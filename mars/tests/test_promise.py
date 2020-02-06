# Copyright 1999-2020 Alibaba Group Holding Ltd.
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

import asyncio
import asyncio.queues
import functools
import sys
import time
import unittest
import weakref

from mars.tests.core import create_actor_pool, aio_case
from mars.utils import build_exc_info
from mars import promise


class ServeActor(promise.PromiseActor):
    def __init__(self):
        super().__init__()
        self._result_list = []

    @promise.reject_on_exception
    async def serve(self, value, delay=None, accept=True, raises=False, callback=None):
        await asyncio.sleep(delay if delay is not None else 0.1)
        if raises:
            raise ValueError('User-induced error')
        self._result_list.append(value)
        if callback:
            await self.tell_promise(callback, value, _accept=accept)

    def get_result(self):
        return self._result_list

    def clear_result(self):
        self._result_list = []


class PromiseTestActor(promise.PromiseActor):
    def __init__(self):
        super().__init__()
        self._finished = False

    def get_finished(self):
        return self._finished

    def reset_finished(self):
        self._finished = False

    async def test_normal(self):
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

    def test_spawn(self, raises=False):
        async def _task():
            await asyncio.sleep(0.5)
            if raises:
                raise SystemError

        ref = self.promise_ref('ServeActor')
        promise.all_([self.spawn_promised(_task) for _ in range(4)]) \
            .then(lambda *_: ref.serve(0, delay=0, _promise=True)) \
            .catch(lambda *exc: ref.serve(exc[0].__name__, delay=0, _promise=True)) \
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

        async def _rejecter(*exc):
            await ref.serve(exc[0].__name__)

        ref.serve(0, delay=2, _timeout=1, _promise=True) \
            .catch(_rejecter) \
            .then(lambda *_: setattr(self, '_finished', True))

    def test_no_timeout(self):
        self._finished = False

        ref = self.promise_ref('ServeActor')

        async def _rejecter(*exc):
            await ref.serve(exc[0].__name__)

        ref.serve(0, delay=1, _timeout=2, _promise=True) \
            .catch(_rejecter) \
            .then(lambda *_: setattr(self, '_finished', True))

    def test_ref_reject(self):
        from mars.errors import WorkerProcessStopped

        self._finished = False
        ref = self.promise_ref('ServeActor')

        async def _rejecter(*exc):
            await ref.serve(exc[0].__name__)

        ref.serve(0, delay=2, _promise=True) \
            .catch(_rejecter) \
            .then(lambda *_: setattr(self, '_finished', True))
        self.reject_promise_refs([ref], *build_exc_info(WorkerProcessStopped))

    def test_addr_reject(self):
        from mars.errors import WorkerDead

        self._finished = False
        ref = self.promise_ref('ServeActor', address=self.address)

        async def _rejecter(*exc):
            await ref.serve(exc[0].__name__)

        ref.serve(0, delay=2, _promise=True) \
            .catch(_rejecter) \
            .then(lambda *_: setattr(self, '_finished', True))
        self.reject_dead_endpoints([self.address], *build_exc_info(WorkerDead))

    def test_closure_refcount(self, content=''):
        ref = self.promise_ref('ServeActor', address=self.address)

        class Intermediate(object):
            def __init__(self, s):
                self.str = s

        new_content = Intermediate('Content: %s' % (content,))

        async def _acceptor(*_):
            await ref.serve(weakref.ref(new_content))
            return 'Processed: %s' % new_content.str

        ref.serve(0, delay=0.5, _promise=True) \
            .then(_acceptor) \
            .then(lambda *_: setattr(self, '_finished', True))


def _raise_exception(exc):
    raise exc


async def wait_test_actor_result(ref, timeout):
    t = time.time()
    while not await ref.get_finished():
        await asyncio.sleep(0.01)
        if time.time() > t + timeout:
            raise TimeoutError


@unittest.skipIf(sys.platform == 'win32', 'does not run in windows')
@aio_case
class Test(unittest.TestCase):
    async def testPromise(self):
        promises = weakref.WeakValueDictionary()
        req_queue = asyncio.queues.Queue()
        value_list = []

        time_unit = 0.1

        async def test_coro_body():
            while True:
                idx, v, success = await req_queue.get()
                if v is None:
                    break
                value_list.append(('thread_body', v))
                await asyncio.sleep(time_unit)
                asyncio.ensure_future(promises[idx].step_next([(v,), dict(_accept=success)]))

        try:
            asyncio.ensure_future(test_coro_body())

            def gen_promise(value, accept=True):
                value_list.append(('gen_promise', value))
                p = promise.Promise()
                promises[p.id] = p
                asyncio.ensure_future(req_queue.put((p.id, value + 1, accept)))
                return p

            # simple promise call
            value_list = []
            p = gen_promise(0) \
                .then(lambda v: gen_promise(v)) \
                .then(lambda v: gen_promise(v))
            await p
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 2), ('thread_body', 3)]
            )

            # continue accepted call with then
            value_list = []
            await p.then(lambda *_: gen_promise(0)) \
                .then(lambda v: gen_promise(v))
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2)]
            )

            # immediate error
            value_list = []
            p = promise.finished() \
                .then(lambda *_: 5 / 0)
            await p.catch(lambda *_: gen_promise(0))
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
            await p
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1)]
            )

            # continue error call
            value_list = []
            p = gen_promise(0) \
                .then(lambda *_: 5 / 0) \
                .then(lambda *_: gen_promise(2))
            await asyncio.sleep(0.5)
            value_list = []
            await p.catch(lambda *_: gen_promise(0))
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1)]
            )

            value_list = []
            await gen_promise(0) \
                .then(lambda v: gen_promise(v).then(lambda x: x + 1)) \
                .then(lambda v: gen_promise(v))
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 3), ('thread_body', 4)]
            )

            value_list = []
            await gen_promise(0) \
                .then(lambda v: gen_promise(v)) \
                .then(lambda v: gen_promise(v, False)) \
                .catch(lambda v: value_list.append(('catch', v)))
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 2), ('thread_body', 3),
                 ('catch', 3)]
            )

            value_list = []
            await gen_promise(0) \
                .then(lambda v: gen_promise(v, False).then(lambda x: x + 1)) \
                .then(lambda v: gen_promise(v)) \
                .catch(lambda v: value_list.append(('catch', v)))
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('catch', 2)]
            )

            value_list = []
            await gen_promise(0) \
                .then(lambda v: gen_promise(v)) \
                .then(lambda v: v + 1) \
                .then(lambda v: gen_promise(v, False)) \
                .catch(lambda v: value_list.append(('catch', v)))
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 3), ('thread_body', 4),
                 ('catch', 4)]
            )

            value_list = []
            await gen_promise(0) \
                .then(lambda v: gen_promise(v)) \
                .catch(lambda v: gen_promise(v)) \
                .catch(lambda v: gen_promise(v)) \
                .then(lambda v: gen_promise(v, False)) \
                .catch(lambda v: value_list.append(('catch', v)))
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 2), ('thread_body', 3),
                 ('catch', 3)]
            )

            value_list = []
            await gen_promise(0) \
                .then(lambda v: gen_promise(v)) \
                .catch(lambda v: gen_promise(v)) \
                .catch(lambda v: gen_promise(v)) \
                .then(lambda v: gen_promise(v, False)) \
                .then(lambda v: gen_promise(v), lambda v: gen_promise(v + 1, False)) \
                .catch(lambda v: value_list.append(('catch', v)))
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 2), ('thread_body', 3),
                 ('gen_promise', 4), ('thread_body', 5),
                 ('catch', 5)]
            )

            value_list = []
            await gen_promise(0) \
                .then(lambda v: gen_promise(v)) \
                .catch(lambda v: gen_promise(v)) \
                .catch(lambda v: gen_promise(v)) \
                .then(lambda v: gen_promise(v, False)) \
                .then(lambda v: gen_promise(v), lambda v: _raise_exception(ValueError)) \
                .catch(lambda *_: value_list.append(('catch',)))
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 2), ('thread_body', 3),
                 ('catch', )]
            )

            value_list = []
            await gen_promise(0) \
                .then(lambda v: gen_promise(v, False)) \
                .catch(lambda v: gen_promise(v, False)) \
                .catch(lambda v: gen_promise(v)) \
                .then(lambda v: gen_promise(v)) \
                .catch(lambda v: value_list.append(('catch', v)))
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('gen_promise', 2), ('thread_body', 3),
                 ('gen_promise', 3), ('thread_body', 4),
                 ('gen_promise', 4), ('thread_body', 5)]
            )

            value_list = []
            await gen_promise(0) \
                .then(lambda v: gen_promise(v, False)) \
                .then(lambda v: gen_promise(v)) \
                .catch(lambda v: value_list.append(('catch', v)))
            self.assertListEqual(
                value_list,
                [('gen_promise', 0), ('thread_body', 1),
                 ('gen_promise', 1), ('thread_body', 2),
                 ('catch', 2)]
            )
        finally:
            self.assertDictEqual(promise._promise_pool, {})

    async def testPromiseActor(self):
        try:
            async with create_actor_pool(n_process=1) as pool:
                serve_ref = await pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = await pool.create_actor(PromiseTestActor)

                await test_ref.test_normal()
                await wait_test_actor_result(test_ref, 10)
                self.assertListEqual(await serve_ref.get_result(), list(range(11)))

                await serve_ref.clear_result()
                await test_ref.reset_finished()

                await test_ref.test_error_raise()
                await wait_test_actor_result(test_ref, 10)
                self.assertListEqual(await serve_ref.get_result(), [-1])
        finally:
            self.assertEqual(promise.get_active_promise_count(), 0)

    async def testAll(self):
        promises = weakref.WeakValueDictionary()
        req_queue = asyncio.queues.Queue()
        value_list = []

        time_unit = 0.1

        async def test_coro_body():
            while True:
                idx, v, success = await req_queue.get()
                if v is None:
                    break
                value_list.append(('thread_body', v))
                await asyncio.sleep(time_unit)
                asyncio.ensure_future(promises[idx].step_next([(v,), dict(_accept=success)]))

        def gen_promise(value, accept=True):
            p = promise.Promise()
            promises[p.id] = p
            asyncio.ensure_future(req_queue.put((p.id, value + 1, accept)))
            return p

        try:
            asyncio.ensure_future(test_coro_body())

            value_list = []
            await promise.all_([]).then(lambda: value_list.append(('all', 0)))
            self.assertListEqual(value_list, [('all', 0)])

            value_list = []
            prior_promises = [gen_promise(idx) for idx in range(4)]
            await promise.all_(prior_promises).then(lambda: value_list.append(('all', 5)))
            del prior_promises

            self.assertListEqual(
                value_list,
                [('thread_body', 1), ('thread_body', 2), ('thread_body', 3),
                 ('thread_body', 4), ('all', 5)]
            )

            value_list = []
            prior_promises = [gen_promise(idx, bool((idx + 1) % 2)) for idx in range(4)]
            await promise.all_(prior_promises).then(
                lambda: value_list.append(('all', 5)),
                lambda *_: value_list.append(('all_catch', 5)),
            )
            del prior_promises

            expected = [('thread_body', 1), ('thread_body', 2), ('thread_body', 3), ('all_catch', 5)]
            self.assertListEqual(value_list[:len(expected)], expected)
            await asyncio.sleep(0.5)

            def _gen_all_promise(*_):
                prior_promises = [gen_promise(idx, bool((idx + 1) % 2)) for idx in range(4)]
                return promise.all_(prior_promises)

            value_list = []
            await gen_promise(0) \
                .then(lambda *_: value_list.append(('pre_all', 0))) \
                .then(_gen_all_promise) \
                .then(lambda v: gen_promise(v)) \
                .then(
                lambda: value_list.append(('all', 5)),
                lambda *_: value_list.append(('all_catch', 5)),
            )
            expected = [('thread_body', 1), ('pre_all', 0),
                        ('thread_body', 1), ('thread_body', 2), ('thread_body', 3),
                        ('all_catch', 5)]
            self.assertListEqual(value_list[:len(expected)], expected)
            await asyncio.sleep(0.5)
        finally:
            self.assertEqual(promise.get_active_promise_count(), 0)

    async def testSpawnPromisedActor(self):
        try:
            async with create_actor_pool(n_process=1) as pool:
                serve_ref = await pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = await pool.create_actor(PromiseTestActor)

                start_time = time.time()
                await test_ref.test_spawn()
                self.assertLess(time.time() - start_time, 0.5)
                await wait_test_actor_result(test_ref, 30)
                self.assertEqual(await serve_ref.get_result(), [0])

                self.assertGreaterEqual(time.time() - start_time, 0.5)
                self.assertLess(time.time() - start_time, 1)

                await serve_ref.clear_result()
                await test_ref.reset_finished()

                start_time = time.time()
                await test_ref.test_spawn(raises=True)
                self.assertLess(time.time() - start_time, 0.5)
                await wait_test_actor_result(test_ref, 30)
                self.assertEqual(await serve_ref.get_result(), ['SystemError'])

                self.assertGreaterEqual(time.time() - start_time, 0.5)
                self.assertLess(time.time() - start_time, 1)
        finally:
            self.assertEqual(promise.get_active_promise_count(), 0)

    async def testAllActor(self):
        try:
            async with create_actor_pool(n_process=1) as pool:
                serve_ref = await pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = await pool.create_actor(PromiseTestActor)

                await test_ref.test_all_promise()

                await wait_test_actor_result(test_ref, 30)
                self.assertListEqual(
                    await serve_ref.get_result(),
                    [-128] + list(range(0, 20, 2)) + list(range(1, 20, 2)) + [127]
                )
        finally:
            self.assertEqual(promise.get_active_promise_count(), 0)

    async def testTimeoutActor(self):
        try:
            async with create_actor_pool(n_process=1) as pool:
                serve_ref = await pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = await pool.create_actor(PromiseTestActor)

                await test_ref.test_timeout()

                await wait_test_actor_result(test_ref, 30)
                self.assertListEqual(await serve_ref.get_result(), [0, 'PromiseTimeout'])
        finally:
            self.assertEqual(promise.get_active_promise_count(), 0)

    async def testNoTimeoutActor(self):
        try:
            async with create_actor_pool(n_process=1) as pool:
                serve_ref = await pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = await pool.create_actor(PromiseTestActor)

                await test_ref.test_no_timeout()

                await wait_test_actor_result(test_ref, 30)

                self.assertListEqual(await serve_ref.get_result(), [0])
        finally:
            self.assertEqual(promise.get_active_promise_count(), 0)

    async def testRefReject(self):
        try:
            async with create_actor_pool(n_process=1) as pool:
                serve_ref = await pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = await pool.create_actor(PromiseTestActor)

                await test_ref.test_ref_reject()

                await wait_test_actor_result(test_ref, 30)
                self.assertListEqual(await serve_ref.get_result(), [0, 'WorkerProcessStopped'])
        finally:
            self.assertEqual(promise.get_active_promise_count(), 0)

    async def testAddrReject(self):
        try:
            async with create_actor_pool(n_process=1) as pool:
                serve_ref = await pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = await pool.create_actor(PromiseTestActor)

                await test_ref.test_addr_reject()

                await wait_test_actor_result(test_ref, 30)
                self.assertListEqual(await serve_ref.get_result(), [0, 'WorkerDead'])
        finally:
            self.assertEqual(promise.get_active_promise_count(), 0)

    async def testClosureRefcount(self):
        try:
            async with create_actor_pool(n_process=1) as pool:
                serve_ref = await pool.create_actor(ServeActor, uid='ServeActor')
                test_ref = await pool.create_actor(PromiseTestActor)

                await test_ref.test_closure_refcount()
                await wait_test_actor_result(test_ref, 30)
                self.assertIsNone((await serve_ref.get_result())[-1]())
        finally:
            self.assertEqual(promise.get_active_promise_count(), 0)
