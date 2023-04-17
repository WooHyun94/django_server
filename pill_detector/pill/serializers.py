from .models import Pill
from rest_framework import serializers

class pill_serializer(serializers.ModelSerializer):
    class Meta:
        model = Pill
        fields = ['name', 'efficacy']