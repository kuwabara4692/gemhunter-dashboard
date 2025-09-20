from flask import Flask, request, render_template, send_file
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap, BoundaryNorm

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        dem_file = request.files['dem']
        dem_path = os.path.join(UPLOAD_FOLDER, dem_file.filename)
        dem_file.save(dem_path)

        output_path = os.path.join(OUTPUT_FOLDER, 'terrain_map.png')
        generate_map(dem_path, output_path)

        return render_template('index.html', download_link='/download')

    return render_template('index.html', download_link=None)

@app.route('/download')
def download():
    return send_file(os.path.join(OUTPUT_FOLDER, 'terrain_map.png'), as_attachment=True)

def generate_map(dem_path, output_path):
    with rasterio.open(dem_path) as dem:
        elevation = dem.read(1)
        transform = dem.transform
        pixel_size_m = transform[0]

    x, y = np.gradient(elevation)
    slope = np.sqrt(x**2 + y**2)
    aspect = np.arctan2(-x, y)

    azimuth = 315
    altitude = 45
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    slope_rad = np.arctan(slope)
    hillshade = (np.sin(alt_rad) * np.cos(slope_rad) +
                 np.cos(alt_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect))
    hillshade = hillshade.clip(0, 1)

    valid_slope = slope[np.isfinite(slope)]
    bounds = np.percentile(valid_slope, [0, 25, 50, 75, 90, 100])
    colors = ['gray', 'lightyellow', 'orange', 'orangered', 'saddlebrown']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    valid_elevation = elevation[np.isfinite(elevation)]
    min_elev = np.percentile(valid_elevation, 5)
    max_elev = np.percentile(valid_elevation, 95)
    contour_levels = np.linspace(min_elev, max_elev, 15)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axs[0].imshow(hillshade, cmap='gray')
    axs[0].contour(elevation, levels=contour_levels, colors='black', linewidths=0.5)
    axs[0].set_title('Hillshade + Contours')
    axs[0].axis('off')
    axs[0].add_artist(ScaleBar(pixel_size_m, units='m', location='lower right'))
    plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    im2 = axs[1].imshow(slope, cmap=cmap, norm=norm)
    axs[1].contour(elevation, levels=contour_levels, colors='black', linewidths=0.5)
    axs[1].set_title('Slope + Contours')
    axs[1].axis('off')
    axs[1].add_artist(ScaleBar(pixel_size_m, units='m', location='lower right'))
    plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04, ticks=bounds)

    im3 = axs[2].imshow(aspect, cmap='twilight')
    axs[2].contour(elevation, levels=contour_levels, colors='black', linewidths=0.5)
    axs[2].set_title('Aspect + Contours')
    axs[2].axis('off')
    axs[2].add_artist(ScaleBar(pixel_size_m, units='m', location='lower right'))
    plt.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
