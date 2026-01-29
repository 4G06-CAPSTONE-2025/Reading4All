from django.core.management.base import BaseCommand
from databases.connect_supabase import get_supabase_client


class Command(BaseCommand):
    help = "Test Supabase connection"

    def handle(self, *args, **options):
        supabase = get_supabase_client()

        # change table name if needed
        response = supabase.table("history").select("*").limit(1).execute()

        self.stdout.write(self.style.SUCCESS("Supabase connection OK"))
        self.stdout.write(str(response.data))
