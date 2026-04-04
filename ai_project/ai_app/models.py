from django.db import models

class Conversation(models.Model):
    title = models.CharField(max_length=200, default="New Chat")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class Message(models.Model):
    ROLE_CHOICES = [("user", "User"), ("bot", "Bot")] # (Database value, Human-readable name) shown in admin/forms
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    source = models.CharField(max_length=500, blank=True)  # stores [SOURCE] citation
    created_at = models.DateTimeField(auto_now_add=True)
    email_sent = models.BooleanField(default=False)

    class Meta:
        ordering = ["created_at"]