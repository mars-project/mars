# Copyright 1999-2018 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import functools
import sys
import threading
import time
import unittest
import weakref
import sys

import gevent

from mars.actors import create_actor_pool
from mars.compat import Queue
from mars import promise


class ServeActor(promise.PromiseActor):
    def __init__(self):
        super(ServeActor, self).__init__()
        self._result_list = []

    def serve(self, value, delay=None, accept=True, callback=None):
        gevent.sleep(delay or 0.1)
        self._result_list.append(value)
        if callback:
            self.tell_promise(callback, value, _accept=accept)

    def get_result(self):
        return self._result_list


class PromiseTestActor(promise.PromiseActor):
    def test_normal(self):
        ref = self.promise_ref('ServeActor')
        p = ref.serve(0, _promise=True)
        for _ in range(10):
            p = p.then(lambda v: ref.serve(v + 1, _promise=True))

    def test_all_promise(self):
        ref = self.promise_ref('ServeActor')
        promises = []

        def subsequent_all(*_):
            def func(idx, *_, **kw):
                return ref.serve(idx, _promise=True, **kw)

            for idx in range(10):
                promises.append(func(idx * 2).then(functools.partial(func, idx * 2 + 1)))
            return promise.all_(promises)

        ref.serve(-128, _promise=True) \
            .then(subsequent_all) \
            .then(lambda *_: ref.serve(127, _promise=True))

    def test_timeout(self):
        ref = self.promise_ref('ServeActor')

        def _rejecter(*exc):
            ref.serve(exc[0].__name__)

        ref.serve(0, delay=2, _timeout=1, _promise=True) \
            .catch(_rejecter)

    def test_no_timeout(self):
        ref = self.promise_ref('ServeActor')

        def _rejecter(*exc):
            ref.serve(exc[0].__name__)

        ref.serve(0, delay=1, _timeout=2, _promise=True) \
            .catch(_rejecter)

    def test_ref_reject(self):
        from mars.errors import WorkerProcessStopped
        try:
            raise WorkerProcessStopped
        except WorkerProcessStopped:
            exc_info = sys.exc_info()

        ref = self.promise_ref('ServeActor')

        def _rejecter(*exc):
            ref.serve(exc[0].__name__)

        ref.serve(0, delay=2, _promise=True) \
            .catch(_rejecter)
        self.reject_promise_ref(ref, *exc_info)


def _raise_exception(exc):
    raise exc


@unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
class Test(unittest.TestCase):
    def testPromise(self):
        promises = weakref.WeakValueDictionary()
        req_queue = Queue()
        value_list = []

        time_unit = 0.1

        def test_thread_body():
            while True:
                idx, v, success = req_queue.get()
                if v is None:
                    break
                value_list.append(('thread_body', v))
                time.sleep(time_unit)
                promises[idx].step_next(v, _accept=success)

        try:
            thread = threading.Thread(target=test_thread_body)
            thread.daemon = True
            thread.start()

            def gen_promise(value, accept=True):
                value_list.append(('gen_promise', value))
                p = promise.Promise()
                promises[p.id] = p
                req_queue.put((p.id, value + 1, accept))
                return p

            value_list = []
            gen_promise(0) \
                .then(lambda v: gen_promise(v)) \
                .then(lambda v: gen_promise(v)) \
                .wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 2), ('thread_body', 3)]
            )

            value_list = []
            gen_promise(0) \
                .then(lambda v: gen_promise(v).then(lambda x: x + 1)) \
                .then(lambda v: gen_promise(v)) \
                .wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 3), ('thread_body', 4)]
            )

            value_list = []
            gen_promise(0) \
                .then(lambda v: gen_promise(v)) \
                .then(lambda v: gen_promise(v, False)) \
                .catch(lambda v: value_list.append(('catch', v))) \
                .wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 2), ('thread_body', 3),
                 ('catch', 3)]
            )

            value_list = []
            gen_promise(0) \
                .then(lambda v: gen_promise(v, False).then(lambda x: x + 1)) \
                .then(lambda v: gen_promise(v)) \
                .catch(lambda v: value_list.append(('catch', v))) \
                .wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('catch', 2)]
            )

            value_list = []
            gen_promise(0) \
                .then(lambda v: gen_promise(v)) \
                .then(lambda v: v + 1) \
                .then(lambda v: gen_promise(v, False)) \
                .catch(lambda v: value_list.append(('catch', v))) \
                .wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 3), ('thread_body', 4),
                 ('catch', 4)]
            )

            value_list = []
            gen_promise(0) \
                .then(lambda v: gen_promise(v)) \
                .catch(lambda v: gen_promise(v)) \
                .catch(lambda v: gen_promise(v)) \
                .then(lambda v: gen_promise(v, False)) \
                .catch(lambda v: value_list.append(('catch', v))) \
                .wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 2), ('thread_body', 3),
                 ('catch', 3)]
            )

            value_list = []
            gen_promise(0) \
                .then(lambda v: gen_promise(v)) \
                .catch(lambda v: gen_promise(v)) \
                .catch(lambda v: gen_promise(v)) \
                .then(lambda v: gen_promise(v, False)) \
                .then(lambda v: gen_promise(v), lambda v: gen_promise(v + 1, False)) \
                .catch(lambda v: value_list.append(('catch', v))) \
                .wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 2), ('thread_body', 3),
                 ('gen_promise', 4), ('thread_body', 5),
                 ('catch', 5)]
            )

            value_list = []
            gen_promise(0) \
                .then(lambda v: gen_promise(v)) \
                .catch(lambda v: gen_promise(v)) \
                .catch(lambda v: gen_promise(v)) \
                .then(lambda v: gen_promise(v, False)) \
                .then(lambda v: gen_promise(v), lambda v: _raise_exception(ValueError)) \
                .catch(lambda *_: value_list.append(('catch',))) \
                .wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 2), ('thread_body', 3),
                 ('catch', )]
            )

            value_list = []
            gen_promise(0) \
                .then(lambda v: gen_promise(v, False)) \
                .catch(lambda v: gen_promise(v, False)) \
                .catch(lambda v: gen_promise(v)) \
                .then(lambda v: gen_promise(v)) \
                .catch(lambda v: value_list.append(('catch', v))) \
                .wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 2), ('thread_body', 3),
                 ('gen_promise', 3), ('thread_body', 4),
                 ('gen_promise', 4), ('thread_body', 5)]
            )

            value_list = []
            gen_promise(0) \
                .then(lambda v: gen_promise(v, False)) \
                .then(lambda v: gen_promise(v)) \
                .catch(lambda v: value_list.append(('catch', v))) \
                .wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('catch', 2)]
            )
        finally:
            gc.collect()
            self.assertDictEqual(promise._promise_pool, {})
            req_queue.put((None, None, None))

    def testPromiseActor(self):
        try:
            with create_actor_pool() as pool:
                serve_ref = pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = pool.create_actor(PromiseTestActor)

                def test_proc():
                    test_ref.test_normal()
                    gevent.sleep(2)
                    self.assertListEqual(serve_ref.get_result(), list(range(11)))

                gl = gevent.spawn(test_proc)
                gl.join()
        finally:
            self.assertDictEqual(promise._promise_pool, {})

    def testAll(self):
        promises = weakref.WeakValueDictionary()
        req_queue = Queue()
        value_list = []

        time_unit = 0.1

        def test_thread_body():
            while True:
                idx, v, success = req_queue.get()
                if v is None:
                    break
                value_list.append(('thread_body', v))
                time.sleep(time_unit)
                promises[idx].step_next(v, _accept=success)

        def gen_promise(value, accept=True):
            p = promise.Promise()
            promises[p.id] = p
            req_queue.put((p.id, value + 1, accept))
            return p

        try:
            thread = threading.Thread(target=test_thread_body)
            thread.daemon = True
            thread.start()

            value_list = []
            promise.all_([]).then(lambda: value_list.append(('all', 0))).wait()
            self.assertListEqual(value_list, [('all', 0)])

            value_list = []
            prior_promises = [gen_promise(idx) for idx in range(4)]
            promise.all_(prior_promises).then(lambda: value_list.append(('all', 5))).wait()
            del prior_promises

            self.assertListEqual(
                value_list,
                [('thread_body', 1), ('thread_body', 2), ('thread_body', 3),
                 ('thread_body', 4), ('all', 5)]
            )

            value_list = []
            prior_promises = [gen_promise(idx, bool((idx + 1) % 2)) for idx in range(4)]
            promise.all_(prior_promises).then(
                lambda: value_list.append(('all', 5)),
                lambda *_: value_list.append(('all_catch', 5)),
            ).wait()
            del prior_promises

            expected = [('thread_body', 1), ('thread_body', 2), ('all_catch', 5)]
            self.assertListEqual(value_list[:len(expected)], expected)
            time.sleep(0.5)

            def _gen_all_promise(*_):
                prior_promises = [gen_promise(idx, bool((idx + 1) % 2)) for idx in range(4)]
                return promise.all_(prior_promises)

            value_list = []
            gen_promise(0) \
                .then(lambda *_: value_list.append(('pre_all', 0))) \
                .then(_gen_all_promise) \
                .then(lambda v: gen_promise(v)) \
                .then(
                lambda: value_list.append(('all', 5)),
                lambda *_: value_list.append(('all_catch', 5)),
            ).wait()
            expected = [('thread_body', 1), ('pre_all', 0), ('thread_body', 1), ('thread_body', 2), ('all_catch', 5)]
            self.assertListEqual(value_list[:len(expected)], expected)
            time.sleep(0.5)
        finally:
            self.assertDictEqual(promise._promise_pool, {})
            req_queue.put((None, None, None))

    def testAllActor(self):
        try:
            with create_actor_pool() as pool:
                serve_ref = pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = pool.create_actor(PromiseTestActor)

                def run_proc_test():
                    test_ref.test_all_promise()
                    gc.collect()
                    gevent.sleep(3)
                    self.assertListEqual(
                        serve_ref.get_result(),
                        [-128] + list(range(0, 20, 2)) + list(range(1, 20, 2)) + [127]
                    )

                gl = gevent.spawn(run_proc_test)
                gl.join()
        finally:
            self.assertDictEqual(promise._promise_pool, {})

    def testTimeoutActor(self):
        try:
            with create_actor_pool() as pool:
                serve_ref = pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = pool.create_actor(PromiseTestActor)

                def run_proc_test():
                    test_ref.test_timeout()
                    gc.collect()
                    gevent.sleep(3)
                    self.assertListEqual(serve_ref.get_result(), [0, 'PromiseTimeout'])

                gl = gevent.spawn(run_proc_test)
                gl.join()
        finally:
            self.assertDictEqual(promise._promise_pool, {})

    def testNoTimeoutActor(self):
        try:
            with create_actor_pool() as pool:
                serve_ref = pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = pool.create_actor(PromiseTestActor)

                def run_proc_test():
                    test_ref.test_no_timeout()
                    gc.collect()
                    gevent.sleep(3)
                    # print(serve_ref.get_result())
                    self.assertListEqual(serve_ref.get_result(), [0])

                gl = gevent.spawn(run_proc_test)
                gl.join()
        finally:
            self.assertDictEqual(promise._promise_pool, {})

    def testRefReject(self):
        try:
            with create_actor_pool() as pool:
                serve_ref = pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = pool.create_actor(PromiseTestActor)

                def run_proc_test():
                    test_ref.test_ref_reject()
                    gc.collect()
                    gevent.sleep(3)
                    self.assertListEqual(serve_ref.get_result(), [0, 'WorkerProcessStopped'])

                gl = gevent.spawn(run_proc_test)
                gl.join()
        finally:
            self.assertDictEqual(promise._promise_pool, {})
