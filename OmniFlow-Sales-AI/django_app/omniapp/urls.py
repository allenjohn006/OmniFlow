"""URL configuration for omniapp."""

from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload, name="upload"),
    path("predict/", views.predict, name="predict"),
    path("train/", views.train_result, name="train_result"),
    path("drift/", views.drift_result, name="drift_result"),
]
