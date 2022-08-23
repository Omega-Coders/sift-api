
from django.urls import path

from . import views

urlpatterns = [
    # path("orb/", views.ORB),
    path("sift/add-test-img/", views.SIFT),
    path("sift/add-ref-img/", views.addRefImg)
]











