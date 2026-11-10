# GlassGS project page

Public project homepage: https://aswhalefall.github.io/GlassGS/

This is a dependency-free static site. GitHub Pages should publish from `main` / `/docs` in **Settings → Pages**.

Edit `index.html`, `styles.css`, and `app.js` in this directory to update the GitHub project page. No build step is needed. Preview from the repository root with:

```sh
python -m http.server 8080 --directory docs
```

Research photos are extracted original images; SVG figures preserve vector labels and the embedded image resolution. `assets/image-provenance.json` records the source and pixel hashes of the four input/output comparisons. The page presents our method, datasets, quantitative evaluation, qualitative comparisons, and limitations. The independent 2D restoration evaluation is clearly distinguished from the 3D reconstruction benchmark.
