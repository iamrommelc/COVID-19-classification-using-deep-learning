from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
# Below here is the code snippet for calculating various performance parameters of our model().
tpr=[]
fpr=[]
total_no_fp=0
total_no_tp=0
total_no_fn=0
total_no_tn=0
total=[]
precision=[]
f1=[]
acc=[]
sp=[]
validation_path=<Enter_testing_imagefolder_path>
for i in range(5):
    valid_datagen = ImageDataGenerator(rescale=1./255)
    valid_dir = valid_datagen.flow_from_directory(validation_path+str(i),
                                                  target_size=(224, 224),
                                                  batch_size=16,
                                                  class_mode='binary',shuffle=False)
                                                  Y_pred = model.predict_generator(valid_dir, steps= len(valid_dir))
                                                  print("\n Y_pred ","\n", Y_pred)
                                                  XtrueX = valid_dir.classes
                                                  y_pred = []
                                                  for it1 in range(len(Y_pred)):
                                                      if Y_pred[it1]>0.8:
                                                          y_pred.append(1)
                                                          if XtrueX[it1]==1:
                                                              total_no_tp = total_no_tp + 1
                                                                  else:
                                                                      total_no_fp = total_no_fp + 1
                                                                          else:
                                                                              y_pred.append(0)
                                                                              if XtrueX[it1] == 1:
                                                                                  total_no_fn = total_no_fn + 1
                                                                                      else:
                                                                                          total_no_tn = total_no_tn + 1
                                                                                      print("case == ", i)
                                                                                      print("valid_dir.classes=",valid_dir.classes)
print("     y_pred      =",y_pred)
Conf_Matrix = confusion_matrix(valid_dir.classes, y_pred)#.ravel()
tpr.append( Conf_Matrix[1][1] /(Conf_Matrix[1][1]+Conf_Matrix[1][0]) ) #tpr=tp/(tp+fn)
fpr.append( Conf_Matrix[0][1] /(Conf_Matrix[0][0]+Conf_Matrix[0][1]) ) #fpr=fp/(tn+fp)
current=len(fpr)-1

for it in range(len(fpr)):
    if len(fpr)==1:
        break
        if fpr[len(fpr)-it-2] <= fpr[current]:
            break
    else:
        temp=fpr[current]
        fpr[current]=fpr[len(fpr)-it-2]
        fpr[len(fpr)-it-2]=fpr[current]
        
        temp=tpr[current]
            tpr[current]=tpr[len(tpr)-it-2]
            tpr[len(tpr)-it-2]=tpr[current]
            
            current=len(tpr)-it-2

pr=  Conf_Matrix[1][1] /(Conf_Matrix[1][1]+Conf_Matrix[0][1])
#pr=tp/(tp+fp)
precision.append(pr)
    f1.append( (2*Conf_Matrix[1][1]) / (2*Conf_Matrix[1][1] + Conf_Matrix[0][1] + Conf_Matrix[1][0])  )
    #f1 = 2*tp / (2*tp+fp+fn)
    acc.append( (Conf_Matrix[1][1]+Conf_Matrix[0][0])/(Conf_Matrix[0][0]+Conf_Matrix[0][1]+Conf_Matrix[1][0]+Conf_Matrix[1][1]) )
    #acc = (tp+tn) / (tp+tn+fp+fn)
    sp.append( Conf_Matrix[0][0] /(Conf_Matrix[0][0]+Conf_Matrix[0][1]) )
#sp = tn / (tn+fp)
print("total false positive = ",total_no_fp,
      "\ntotal  true positive = ",total_no_tp,
      "\ntotal false negative = ",total_no_fn,
      "\ntotal  true negative = ",total_no_tn)

accuarcy=(total_no_tp+total_no_tn)/(total_no_tp+total_no_tn+total_no_fn+total_no_fp)
precision=(total_tp)/(total_tp+total_fp)
recall=(total_tp)/(total_tp+total_fn)
f1_score=(2.0*precision*recall)/(precision+recall)
iou=(total_tp)/(total_tp+total_fp+total_fn)

print("\nAccuracy = ",accuracy)
print("\nPrecision = ",precision)
print("\nRecall  = ",recall)
print("\nF1_score = ",f1_score)
print("\nIoU = ", iou)
