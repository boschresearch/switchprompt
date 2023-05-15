""" Utility classes and functions related to SwitchPrompt (EACL 2023).
Copyright (c) 2022 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import sys

import evaluate
from evaluate.utils import launch_gradio_widget


sys.path = [p for p in sys.path if p != "/home/user/app"]
module = evaluate.load("seqeval")
sys.path = ["/home/user/app"] + sys.path

launch_gradio_widget(module)
