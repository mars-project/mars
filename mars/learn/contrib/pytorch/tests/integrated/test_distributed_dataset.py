# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import mars.tensor as mt
from mars.learn.tests.integrated.base import LearnIntegrationTestBase
from mars.learn.contrib.pytorch.dataset import MarsTorchDataset
from mars.context import DistributedContext
from mars.session import new_session
from mars.utils import lazy_import

torch_installed = lazy_import('torch', globals=globals()) is not None


@unittest.skipIf(not torch_installed, 'pytorch not installed')
class Test(LearnIntegrationTestBase):
    def testDistributedRunPyTorchScript(self):
        service_ep = 'http://127.0.0.1:' + self.web_port
        scheduler_ep = '127.0.0.1:' + self.scheduler_port
        with new_session(service_ep) as sess:
            raw = np.random.rand(100, 200)
            data = mt.tensor(raw, chunk_size=40)
            data.execute(name='data', session=sess)

            with DistributedContext(scheduler_address=scheduler_ep, session_id=sess.session_id):
                t = mt.named_tensor('data')
                dataset = MarsTorchDataset(t)
                self.assertEqual(len(dataset), 100)

                sample = [2, 5, 7, 9, 10]
                r1 = dataset[sample][0]
                np.testing.assert_array_equal(raw[sample], r1)
