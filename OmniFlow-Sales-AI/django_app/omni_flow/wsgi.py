"""WSGI config for OmniFlow Django project."""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "omni_flow.settings")
application = get_wsgi_application()
