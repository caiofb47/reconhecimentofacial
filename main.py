import cv2
import numpy as np
from matplotlib import pyplot as plt

#Função para redimensionar uma imagem
def redim(img, largura): 
    alt = int(img.shape[0]/img.shape[1]*largura)
    img = cv2.resize(img, (largura, alt), interpolation =
    cv2.INTER_AREA)
    return img

#Cria o detector de faces baseado no XML
df = cv2.CascadeClassifier("modelo/haarcascade_frontalface_default.xml")

#Abre o vídeo
video_lido = cv2.VideoCapture("Videos/video2.mp4") # se passar 0 é a webcam


# Contador que serve de controle para nome das imagens salvas
count = 0

# Loop main :D
while True:
    #read() retorna 1-Se houve sucesso e 2-O próprio frame
    (sucesso, frame) = video_lido.read() # Função com dois retornos
    if not sucesso: #final do vídeo
        break

    #reduz tamanho do frame para acelerar processamento
    frame = redim(frame, 320)
    #converte para tons de cinza
    frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detecta as faces no frame
    faces = df.detectMultiScale(frame_pb, scaleFactor = 1.1, minNeighbors=3, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)
    frame_temp = frame.copy()

    #histograma geral
    hist_todos = cv2.calcHist([frame_temp],[0],None,[256],[0,256])
    #plt.plot(hist_todos)
    #plt.savefig("Hist/histo"+str(count)+".png")
    #plt.clf() # Limpa o gráfico
        
    # FIM HISTOGRAMA
    imCom = cv2.imread('Pessoas/imgComparacao.png') #imgComparacao.png

    # Cria os retângulos
    for (x, y, lar, alt) in faces:
        imgS = cv2.rectangle(frame_temp, (x, y), (x + lar, y + alt), (0,255, 255), 2) # RGB
        
        # Salva cada frame com rosto
        rec = imgS[y:y+alt,x:x+lar]
        cv2.imwrite("Pessoas/image"+str(count)+".png",rec) # se não converter o count para string da pau
        
        
        #methods = ['cv2.TM_CCOEFF_NORMED']
        
        # Aplicando o template Matching
        res = cv2.matchTemplate(imCom,imgS,cv2.TM_CCOEFF_NORMED)
        
        #Recupera a similaridade entre o template e o conteúdo da Imagem de busca
        min_val, similaridade, min_loc, max_loc = cv2.minMaxLoc(res)
        texto = 'Similaridade com {0} entre Imagens é {1}%'.format(cv2.TM_CCOEFF_NORMED,round(similaridade*100,2))
        
        if similaridade > 0.85:
            #plt.subplot(121), plt.imshow(imgS, cmap='gray')
            #plt.title('Template'), plt.xticks([]), plt.yticks([])
            #plt.subplot(122), plt.imshow(imCom, cmap='gray')
            #plt.title('Imagem de Busca'), plt.xticks([]), plt.yticks([])
            #plt.suptitle(cv2.TM_CCOEFF_NORMED)
            cv2.imwrite("Pessoas/Compare/image"+str(count)+".png",imgS)
            #exit()
        
        # Média
        im = cv2.blur(rec, (alt,lar))#
        cv2.imwrite("Pessoas/Media/image"+str(count)+".png",im)
        
        # Mediana
        img2 = np.vstack([
               np.hstack([rec,
               cv2.medianBlur(rec, 3)]),
               np.hstack([cv2.medianBlur(rec, 5),
               cv2.medianBlur(rec, 7)]),
               np.hstack([cv2.medianBlur(rec, 9),
               cv2.medianBlur(rec, 11)]),
        ])
        cv2.imwrite("Pessoas/Mediana/image"+str(count)+".png",img2)
        # FIM SALVA

        # HISTOGRAMA
        hist_full = cv2.calcHist([imgS],[0],None,[256],[0,256])
        #plt.plot(hist_full)
        #plt.savefig("HistFace/histo"+str(count)+".png")
        #plt.clf() # Limpa o gráfico
        
    # FIM HISTOGRAMA
    count += 1

    #Exibe um frame redimensionado (com perca de qualidade)
    #cv2.imshow("Encontrando faces...", redim(frame_temp, 640))

    #Espera que a tecla 's' seja pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break


#fecha streaming
video_lido.release()
cv2.destroyAllWindows()
