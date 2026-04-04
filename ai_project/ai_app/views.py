from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from .models import Conversation, Message
from .rag.langchain_rag_pipeline import stream_question
from .agent import send_rag_email
import json

def home(request):
    conversations = Conversation.objects.order_by("-created_at")[:30]
    return render(request, "home.html", {"conversations": conversations})

@require_POST
def new_conversation(request):
    conv = Conversation.objects.create(title="New Chat")
    return JsonResponse({"id": conv.id, "title": conv.title})

@require_GET
def get_conversation(request, conv_id):
    try:
        conv = Conversation.objects.get(id=conv_id)
        messages = list(conv.messages.values("id", "role", "content", "source"))
        return JsonResponse({"id": conv.id, "title": conv.title, "messages": messages})
    except Conversation.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

def stream_rag(request):
    question = request.GET.get("question", "").strip()
    conv_id = request.GET.get("conv_id", "").strip()

    if not question:
        def empty():
            yield "No question provided."
        return StreamingHttpResponse(empty(), content_type="text/plain")

    try:
        conv = Conversation.objects.get(id=conv_id)
    except (Conversation.DoesNotExist, ValueError):
        conv = Conversation.objects.create(title=question[:60])

    Message.objects.create(conversation=conv, role="user", content=question)

    if conv.title == "New Chat":
        conv.title = question[:60]
        conv.save()

    collected = {"text": ""}

    def generate():
        for chunk in stream_question(question):
            collected["text"] += chunk
            yield chunk

        marker = "\n\n[SOURCE]"
        idx = collected["text"].find(marker)
        answer = collected["text"]
        source = ""
        if idx != -1:
            answer = collected["text"][:idx]
            source = collected["text"][idx + len(marker):].strip()

        # Save bot message with id so frontend can reference it for email
        msg = Message.objects.create(conversation=conv, role="bot", content=answer, source=source)
        yield f"\n\n[CONV_ID] {conv.id}"
        yield f"\n\n[MSG_ID] {msg.id}"  # send message id for email action

    response = StreamingHttpResponse(generate(), content_type="text/plain")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response

@require_POST
def send_email_report(request):
    """Agent endpoint: send last bot message by msg_id to an email."""
    try:
        data = json.loads(request.body)
        msg_id = data.get("msg_id")
        to_email = data.get("email", "").strip()

        if not to_email or not msg_id:
            return JsonResponse({"success": False, "error": "Missing email or msg_id"})

        msg = Message.objects.get(id=msg_id, role="bot")
        # Get the user question (previous message in same conversation)
        prev = Message.objects.filter(
            conversation=msg.conversation,
            role="user",
            created_at__lt=msg.created_at
        ).last()
        question = prev.content if prev else "Query"

        result = send_rag_email(to_email, question, msg.content, msg.source)

        if result["success"]:
            msg.email_sent = True
            msg.save()

        return JsonResponse(result)
    except Message.DoesNotExist:
        return JsonResponse({"success": False, "error": "Message not found"})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})