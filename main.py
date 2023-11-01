import cv2
import os


def segment_apples(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se puede cargar la imagen {image_path}")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    # Calcular el histograma de color en el componente Hue
    hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    # Aplicar la retroproyección usando el histograma de color
    backproj = cv2.calcBackProject([hue], [0], hist, [0, 180], scale=1)
    # Aplicar la detección de bordes usando derivadas de Sobel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # Combinar la retroproyección y la detección de bordes
    combined = cv2.bitwise_and(backproj, edges)
    output_path = os.path.join("output", os.path.basename(image_path))
    cv2.imwrite(output_path, combined)
    print(f"Imagen segmentada guardada en {output_path}")
    

def main():
    input_directory = "imgs"
    output_directory = "output"
    os.makedirs(output_directory, exist_ok=True)

    # Procesar todas las imágenes en el directorio de entrada
    for filename in os.listdir(input_directory):
        if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            image_path = os.path.join(input_directory, filename)
            segment_apples(image_path)
            

if __name__=='__main__':
    main()
