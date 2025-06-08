# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import unittest
from core import router

class TestRouter(unittest.TestCase):

    def test_non_tool_input(self):
        input_text = "Hello, how are you?"
        response = router.handle_user_input(input_text)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_valid_tool_command(self):
        input_text = '/generate_image "cat in space"'
        response = router.handle_user_input(input_text)
        self.assertIsInstance(response, str)
        self.assertTrue("cat" in response.lower() or "image" in response.lower())

    def test_invalid_tool_command(self):
        input_text = '/nonexistent_tool "test"'
        response = router.handle_user_input(input_text)
        self.assertIsInstance(response, str)
        self.assertIn("not found", response.lower())

if __name__ == "__main__":
    unittest.main()
