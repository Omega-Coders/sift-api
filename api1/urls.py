
from django.urls import path

from . import views

urlpatterns = [
    path("orb/", views.ORB),
    path("sift/", views.SIFT),
    path("surf/", views.SURF)
]











