from django.apps import AppConfig

class AiAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ai_app'

    def ready(self):
        from .rag.langchain_rag_pipeline import initialize_rag
        initialize_rag()