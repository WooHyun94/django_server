from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import JsonResponse

from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView

from rest_framework import viewsets
from .models import Pill
from .serializers import pill_serializer

import io

from roboflow import Roboflow
# from ultralytics import YOLO
from pathlib import Path

from django.conf import settings

pill_dict = { 0 : 'bearse', 
1 : 'doctor_bearse',
2 : 'festal_gold',
3 : 'festal_plus',
4 : 'panpyrin_t',
5 : 'tylenol'
}

# Create your views here.
class pill_viewset(viewsets.ModelViewSet):
    queryset = Pill.objects.all()
    serializer_class = pill_serializer

class PillDetectorAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser, JSONParser,)
    serializer_class = None
    model_class = None

    def get_queryset(self):
        pass

    def post(self, request, *args, **kwargs):
        data = request.data
        print(data["pill_image"])
        image_path = default_storage.save(
            f'temporary_storage/{data["pill_image"].name}', 
            data['pill_image']
        )
        
        full_image_path = str(Path(settings.MEDIA_ROOT) / image_path)
        # model = YOLO('pill_param.pt')

        rf = Roboflow(api_key="t4OKbq2h3h6wTIA2Sj8u")
        project = rf.workspace().project("pill_detector")
        model = project.version(1).model
        
        #results = model(full_image_path)
        results = model.predict(full_image_path, confidence=40, overlap=30).json()
        
        #pill_list = list(map(int, results[0].boxes.cls.tolist()))
        pill_list = [x['class'] for x in results['predictions']]
        
        pill_set = set(pill_list)

        res_list = list()
        
        # for pill in pill_set:
        #     res_list.append({'name' : Pill.objects.get(name=pill_dict[pill]).name,
        #                      'efficacy' : Pill.objects.get(name=pill_dict[pill]).efficacy})
        for pill in pill_set:
            res_list.append({'name' : Pill.objects.get(name=pill).name,
                             'efficacy' : Pill.objects.get(name=pill).efficacy})
        print(f'pill name list : {res_list}')
        print(f'pill list : {pill_list}')
        print(f'pill set : {pill_set}')
        

        default_storage.delete(full_image_path)

        #return Response(status=status.HTTP_200_OK)
        return JsonResponse({'data' : res_list})
    

def pill_detector_test(request):
    return render(request, 'pill/container/pill_detector_test.html')