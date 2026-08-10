window.MathJax = {
  loader: { load: ["[tex]/physics"] },
  tex: {
    packages: { "[+]": ["physics"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};
