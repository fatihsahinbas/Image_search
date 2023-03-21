import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import json
import os

app = Flask(__name__)

# Görüntü özelliklerinin okunduğu yer. Dokümanda da belirttiğim FeatureExtractor her şeyin başladığı yer.
fe = FeatureExtractor()
features = []
img_paths = []

# Burası offline.py dosyasını çalıştırdıktan sonra feaure klasörüne atılan npy uzantılı dosyaları for döngüsünde tarayıp bunları features değişkenine atıyor.
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)

# Burası yaptığımız index.html dosyasını çalıştırmamız için gerekli fonksiyon
@app.route('/', methods=['GET', 'POST'])
def index():
    #eklediğimiz imaj dosyasını post ediyoruz.
    if request.method == 'POST':
        file = request.files['query_img']
        idssize= request.form.get('ids_size')

        # Sorgu görüntüsünü kaydedeiyoruz.
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Arama işlemini başlatıyoruz.
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:int(idssize)]  # Top 3 results
        scores = [(dists[id], img_paths[id],ids.size) for id in ids]

        #apimiz için results diye bir list değişkeni yaratıyoruz.
        results = []

        #burada hangi alanları ve özellikleri apiye eklemek istiyorsak onları ekliyoruz.
        for item in scores:
            results.append({
                "filename" : os.path.join(item[1]),
                "uncertainty": os.path.join(str(item[0])),
                "size": os.path.join(str(item[2]))
            })

        # Sonuçlarla birlikte json dosyası oluşturuyoruz.
        with open('data.json', 'w') as outputfile:
            json.dump(results, outputfile, ensure_ascii=False, indent=4)        

        # query_path,scores,ids_size ların buradan alınıp index.html e basılması oluşturuluyor ve render ediliyor.
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores, ids_size=idssize)
    else:
        # hiçbir veri olmadığında sadece index.html'in saf hali geliyor.
        return render_template('index.html')

# burasını main metot diye tanımlayabiliriz. app.run çalışmasını sağlıyor.
if __name__=="__main__":
    app.run("0.0.0.0")