"""URL configuration for omniapp."""

from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload, name="upload"),
    path("upload/status/<str:job_id>/", views.upload_status, name="upload_status"),
    path("predict/", views.predict, name="predict"),
    path("train/", views.train_result, name="train_result"),
    path("drift/", views.drift_result, name="drift_result"),
    path("drift/status/<str:job_id>/", views.drift_status, name="drift_status"),
]
