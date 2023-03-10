import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import json
import os

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []

for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        idssize= request.form.get('ids_size')

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:int(idssize)]  # Top 3 results
        scores = [(dists[id], img_paths[id],ids.size) for id in ids]

        results = []

        for item in scores:
            results.append({
                "filename" : os.path.join(item[1]),
                "uncertainty": os.path.join(str(item[0])),
                "size": os.path.join(str(item[2]))
            })

        # Create a JSON file with the results
        with open('data.json', 'w') as outputfile:
            json.dump(results, outputfile, ensure_ascii=False, indent=4)        

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores, ids_size=idssize)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")