from django.shortcuts import render
from django.http import HttpResponse

from rest_framework.response import Response

from rest_framework.decorators import api_view
from rest_framework.decorators import parser_classes
from rest_framework.parsers import JSONParser

import numpy as np
import cv2 as cv
from PIL import Image
import io
from base64 import b64decode, b64encode


@api_view(['POST'])
@parser_classes([JSONParser])
def ORB(request):
        # RefImg = request.data.image1
        # QueryImg = request.data.image2


        # data = JSONParser().parse(request)
        # serializer = ImageSerializer(data)
        # new_data = serializer.data['im1']+serializer.data['im2']


        base64_data1 = request.data.get("im1", None).split(',', 1)[1]
        base64_data2 = request.data.get("im2", None).split(',', 1)[1]
        data = b64decode(base64_data1)
        data2 = b64decode(base64_data2)
        pimg = Image.open(io.BytesIO(data))
        pimg2 = Image.open(io.BytesIO(data2))


        training_image = cv.cvtColor(np.array(pimg), cv.COLOR_BGR2RGB)
        test_image = cv.cvtColor(np.array(pimg2), cv.COLOR_BGR2RGB)

        

        training_gray = cv.cvtColor(training_image, cv.COLOR_RGB2GRAY)
        test_gray = cv.cvtColor(test_image, cv.COLOR_RGB2GRAY)

        h, w = training_gray.shape

        MIN_MATCH_COUNT = 10

        orb = cv.ORB_create()

        kp1, des1 = orb.detectAndCompute(training_image,None)
        kp2, des2 =  orb.detectAndCompute(test_image,None)
        FLANN_INDEX_KDTREE = 1

        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches=flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32), 2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = training_gray.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts,M)
            
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None
        
        imm = test_image.copy()

        matrix = cv.getPerspectiveTransform(dst, pts)

        destination_image = cv.warpPerspective(imm, matrix, (w, h))
        
        img = Image.fromarray(destination_image, 'RGB')
        si = img.resize((300,300),Image.ANTIALIAS)
        buffered = io.BytesIO()
        si.save(buffered, format="JPEG")
        img_str = b64encode(buffered.getvalue())

        final_output = {
            "Transformed Image": img_str
        }

        return Response(final_output)




