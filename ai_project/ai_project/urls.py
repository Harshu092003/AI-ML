"""
URL configuration for ai_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from ai_app import views
urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.home, name="home"),
    path("stream/", views.stream_rag, name="stream_rag"),
    path("conversation/new/", views.new_conversation, name="new_conversation"),
    path("conversation/<int:conv_id>/", views.get_conversation, name="get_conversation"),
    path("send-email/", views.send_email_report, name="send_email"),
]
