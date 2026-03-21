# Farouq Oguntoye Personal Site

Source for [Herr-Professor.github.io](https://Herr-Professor.github.io), a Jekyll-based personal site focused on projects, experiments, experience, and a downloadable resume.

## Site structure

- Home page in `_pages/about.md`
- Projects in `_portfolio/`
- Experience in `_teaching/`
- Experiments in `_talks/`
- Resume in `_pages/cv.md`
- Blog index in `_pages/year-archive.html`
- Static assets and downloads in `images/` and `files/`

## Running locally

1. Install Ruby, Bundler, and Node.js.
2. Install gems with `bundle install`.
3. Start the site with `bundle exec jekyll serve -l -H localhost`.
4. Open `http://localhost:4000`.

If your local Ruby setup is awkward, use Docker instead:

```bash
docker compose up
```

## Deployment

GitHub Pages builds this repository from the default branch. The production URL is `https://Herr-Professor.github.io`.
