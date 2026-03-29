import os
import sys

# Set environment variables to disable optional database drivers
os.environ.setdefault('DISABLE_FALKORDB', '1')
os.environ.setdefault('DISABLE_KUZU', '1')
os.environ.setdefault('DISABLE_NEPTUNE', '1')

# Add project root to Python path for imports (FIRST priority)
# This ensures local graphiti_core is used instead of installed package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(0, project_root)
