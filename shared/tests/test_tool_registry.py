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
import sys, os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tools import tool_registry

class TestToolRegistry(unittest.TestCase):

    def test_get_existing_tool(self):
        tool = tool_registry.get_tool("generate_image")
        self.assertIsNotNone(tool)
        self.assertTrue(callable(tool))

    def test_get_nonexistent_tool(self):
        tool = tool_registry.get_tool("unknown_tool")
        self.assertIsNone(tool)

    def test_list_tools(self):
        tools = tool_registry.list_tools()
        self.assertIsInstance(tools, list)
        self.assertIn("generate_image", tools)

    def test_describe_tools(self):
        desc = tool_registry.describe_tools()
        self.assertIsInstance(desc, dict)
        self.assertIn("generate_image", desc)
        self.assertIn("Creates an image", desc["generate_image"])

if __name__ == "__main__":
    unittest.main()

