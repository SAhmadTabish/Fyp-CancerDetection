from django.shortcuts import render, redirect
from .forms import MedicalImageForm
from .models import MedicalImage
from django.conf import settings
import cv2
import numpy as np
import os

def upload_image(request):
    if request.method == 'POST':
        form = MedicalImageForm(request.POST, request.FILES)
        if form.is_valid():
            medical_image = form.save()
            process_image(medical_image.image.path)
            return redirect('image_list')
    else:
        form = MedicalImageForm()
    return render(request, 'preprocessing/upload_image.html', {'form': form})

def process_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Noise reduction
    denoised_image = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)

    # Image normalization
    normalized_image = cv2.normalize(denoised_image, None, 0, 255, cv2.NORM_MINMAX)

    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(normalized_image)

    # Save the processed image
    processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed_images', os.path.basename(image_path))
    cv2.imwrite(processed_image_path, enhanced_image)

def image_list(request):
    images = MedicalImage.objects.all()
    return render(request, 'preprocessing/image_list.html', {'images': images})
