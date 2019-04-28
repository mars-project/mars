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

from mars.actors import create_actor_pool
from mars.compat import Queue, TimeoutError
from mars.utils import build_exc_info
from mars import promise


class ServeActor(promise.PromiseActor):
    def __init__(self):
        super(ServeActor, self).__init__()
        self._result_list = []

    @promise.reject_on_exception
    def serve(self, value, delay=None, accept=True, raises=False, callback=None):
        self.ctx.sleep(delay or 0.1)
        if raises:
            raise ValueError('User-induced error')
        self._result_list.append(value)
        if callback:
            self.tell_promise(callback, value, _accept=accept)

    def get_result(self):
        return self._result_list

    def clear_result(self):
        self._result_list = []


class PromiseTestActor(promise.PromiseActor):
    def __init__(self):
        super(PromiseTestActor, self).__init__()
        self._finished = False

    def get_finished(self):
        return self._finished

    def test_normal(self):
        self._finished = False

        assert self.promise_ref().uid == self.uid

        ref = self.promise_ref('ServeActor')
        assert ref.__getattr__('_caller') is self

        p = ref.serve(0, _promise=True)

        ref = self.promise_ref(self.ctx.actor_ref('ServeActor'))
        for _ in range(10):
            p = p.then(lambda v: ref.serve(v + 1, _promise=True))
        p.then(lambda *_: setattr(self, '_finished', True))

    def test_error_raise(self):
        self._finished = False

        ref = self.promise_ref('ServeActor')
        ref.serve(0, raises=True, _promise=True) \
            .then(lambda v: ref.serve(v + 1, _promise=True)) \
            .catch(lambda *_: ref.serve(-1, _promise=True)) \
            .then(lambda *_: setattr(self, '_finished', True))

    def test_all_promise(self):
        self._finished = False

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
            .then(lambda *_: ref.serve(127, _promise=True)) \
            .then(lambda *_: setattr(self, '_finished', True))

    def test_timeout(self):
        self._finished = False

        ref = self.promise_ref('ServeActor')

        def _rejecter(*exc):
            ref.serve(exc[0].__name__)

        ref.serve(0, delay=2, _timeout=1, _promise=True) \
            .catch(_rejecter) \
            .then(lambda *_: setattr(self, '_finished', True))

    def test_no_timeout(self):
        self._finished = False

        ref = self.promise_ref('ServeActor')

        def _rejecter(*exc):
            ref.serve(exc[0].__name__)

        ref.serve(0, delay=1, _timeout=2, _promise=True) \
            .catch(_rejecter) \
            .then(lambda *_: setattr(self, '_finished', True))

    def test_ref_reject(self):
        from mars.errors import WorkerProcessStopped

        self._finished = False
        ref = self.promise_ref('ServeActor')

        def _rejecter(*exc):
            ref.serve(exc[0].__name__)

        ref.serve(0, delay=2, _promise=True) \
            .catch(_rejecter) \
            .then(lambda *_: setattr(self, '_finished', True))
        self.reject_promise_refs([ref], *build_exc_info(WorkerProcessStopped))

    def test_addr_reject(self):
        from mars.errors import WorkerDead

        self._finished = False
        ref = self.promise_ref('ServeActor', address=self.address)

        def _rejecter(*exc):
            ref.serve(exc[0].__name__)

        ref.serve(0, delay=2, _promise=True) \
            .catch(_rejecter) \
            .then(lambda *_: setattr(self, '_finished', True))
        self.reject_dead_endpoints([self.address], *build_exc_info(WorkerDead))


def _raise_exception(exc):
    raise exc


def wait_test_actor_result(ref, timeout):
    import gevent
    t = time.time()
    while not ref.get_finished():
        gevent.sleep(0.1)
        if time.time() > t + timeout:
            raise TimeoutError


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

            # simple promise call
            value_list = []
            p = gen_promise(0) \
                .then(lambda v: gen_promise(v)) \
                .then(lambda v: gen_promise(v))
            p.wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 2), ('thread_body', 3)]
            )

            # continue accepted call with then
            value_list = []
            p.then(lambda *_: gen_promise(0)) \
                .then(lambda v: gen_promise(v)) \
                .wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2)]
            )

            # immediate error
            value_list = []
            p = promise.finished() \
                .then(lambda *_: 5 / 0)
            p.catch(lambda *_: gen_promise(0)) \
                .wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1)]
            )

            # chained errors
            value_list = []
            p = promise.finished(_accept=False) \
                .catch(lambda *_: 1 / 0) \
                .catch(lambda *_: 2 / 0) \
                .catch(lambda *_: gen_promise(0)) \
                .catch(lambda *_: gen_promise(1))
            p.wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1)]
            )

            # continue error call
            value_list = []
            p = gen_promise(0) \
                .then(lambda *_: 5 / 0) \
                .then(lambda *_: gen_promise(2))
            time.sleep(0.5)
            value_list = []
            p.catch(lambda *_: gen_promise(0)) \
                .wait()
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1)]
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
            with create_actor_pool(n_process=1) as pool:
                serve_ref = pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = pool.create_actor(PromiseTestActor)

                test_ref.test_normal()
                wait_test_actor_result(test_ref, 10)
                self.assertListEqual(serve_ref.get_result(), list(range(11)))

                serve_ref.clear_result()

                test_ref.test_error_raise()
                wait_test_actor_result(test_ref, 10)
                self.assertListEqual(serve_ref.get_result(), [-1])
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
            with create_actor_pool(n_process=1) as pool:
                serve_ref = pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = pool.create_actor(PromiseTestActor)

                test_ref.test_all_promise()
                gc.collect()
                wait_test_actor_result(test_ref, 30)
                self.assertListEqual(
                    serve_ref.get_result(),
                    [-128] + list(range(0, 20, 2)) + list(range(1, 20, 2)) + [127]
                )
        finally:
            self.assertDictEqual(promise._promise_pool, {})

    def testTimeoutActor(self):
        try:
            with create_actor_pool(n_process=1) as pool:
                serve_ref = pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = pool.create_actor(PromiseTestActor)

                test_ref.test_timeout()
                gc.collect()
                wait_test_actor_result(test_ref, 30)
                self.assertListEqual(serve_ref.get_result(), [0, 'PromiseTimeout'])
        finally:
            self.assertDictEqual(promise._promise_pool, {})

    def testNoTimeoutActor(self):
        try:
            with create_actor_pool(n_process=1) as pool:
                serve_ref = pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = pool.create_actor(PromiseTestActor)

                test_ref.test_no_timeout()
                gc.collect()
                wait_test_actor_result(test_ref, 30)

                self.assertListEqual(serve_ref.get_result(), [0])
        finally:
            self.assertDictEqual(promise._promise_pool, {})

    def testRefReject(self):
        try:
            with create_actor_pool(n_process=1) as pool:
                serve_ref = pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = pool.create_actor(PromiseTestActor)

                test_ref.test_ref_reject()
                gc.collect()
                wait_test_actor_result(test_ref, 30)
                self.assertListEqual(serve_ref.get_result(), [0, 'WorkerProcessStopped'])
        finally:
            self.assertDictEqual(promise._promise_pool, {})

    def testAddrReject(self):
        try:
            with create_actor_pool(n_process=1) as pool:
                serve_ref = pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = pool.create_actor(PromiseTestActor)

                test_ref.test_addr_reject()
                gc.collect()
                wait_test_actor_result(test_ref, 30)
                self.assertListEqual(serve_ref.get_result(), [0, 'WorkerDead'])
        finally:
            self.assertDictEqual(promise._promise_pool, {})
