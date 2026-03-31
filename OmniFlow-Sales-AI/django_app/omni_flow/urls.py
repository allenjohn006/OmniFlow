"""Root URL configuration for OmniFlow Django project."""

from django.urls import path, include

urlpatterns = [
    path("", include("omniapp.urls")),
]
