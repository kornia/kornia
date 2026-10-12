The documentation site was rebuilt on `pydata-sphinx-theme` with a top navigation bar
(Learn / API / Models plus Ecosystem, About and Support menus), a redesigned landing page, a
restructured API reference with per-topic subpages, and a long list of fixed doc examples and
removed dead interactive demos. Old deep links into the split module pages
(e.g. `augmentation.module.html#kornia.augmentation.RandomAffine`) are forwarded to the subpage
that now documents the object, so existing links keep resolving. The furo layout stayed available
behind `KORNIA_DOCS_THEME=furo` until #4304 removed that fallback. (#4155)
