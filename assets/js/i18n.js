(() => {
  'use strict';
  const key = 'co5177-language';
  const controls = [...document.querySelectorAll('[data-language]')];
  const normalize = text => text.trim().replace(/\s+/g, ' ');
  const dictionaryUrl = new URL('assets/js/i18n-data.json', document.baseURI);
  let selected = 'vi';
  try { selected = localStorage.getItem(key) || 'vi'; } catch (_) { /* storage can be unavailable */ }
  if (!['en', 'vi'].includes(selected)) selected = 'vi';
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent || !node.nodeValue.trim() || parent.closest('script,style,code,pre,svg,.js-plotly-plot,.language-switch')) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    }
  });
  const nodes = [];
  while (walker.nextNode()) {
    const node = walker.currentNode;
    nodes.push({node, raw: node.nodeValue, normalized: normalize(node.nodeValue)});
  }
  const attributes = [...document.querySelectorAll('[aria-label],img[alt]')].flatMap(element => ['aria-label', 'alt'].filter(name => element.hasAttribute(name)).map(name => ({element, name, original: element.getAttribute(name)})));
  const originalTitle = document.title;
  let chartSource = null;
  if (typeof edaCharts !== 'undefined' && window.Plotly) chartSource = JSON.parse(JSON.stringify(edaCharts));
  const chartTitles = [
    'Số bình luận theo tập','Phân phối hai nhãn nhị phân','Phân phối đồng thời hai nhãn','Số bình luận theo chủ đề','Tỷ lệ độc hại theo chủ đề',
    'Số đơn vị khoảng trắng (đến phân vị 99%: 163)','Số ký tự (đến phân vị 99%: 700)','20 stopword minh họa thường gặp nhất','25 đơn vị thường gặp sau lọc','Số đơn vị khác nhau theo nhãn độc hại',
    'Tần suất đơn vị — Không độc hại (0)','Tần suất đơn vị — Độc hại (1)','TF-IDF trung bình — Không độc hại (0)','TF-IDF trung bình — Độc hại (1)','20 cặp liền kề thường gặp nhất',
    'Cosine giữa vector TF-IDF trung bình của chủ đề','Số đơn vị khác nhau theo chủ đề','Tần suất — GiaiTri','TF-IDF trung bình — GiaiTri','Số đơn vị khoảng trắng theo nhãn',
    'Số ký tự và đơn vị khoảng trắng','Tỷ lệ bình luận có đặc điểm hình thức','Tỷ lệ đơn vị khác nhau trong bình luận','TF-IDF bigram trung bình — Không độc hại','TF-IDF bigram trung bình — Độc hại','Phân phối số từ tiếng Việt theo bình luận'
  ];
  const plotLabels = {
    'Comments':'Bình luận','Label value':'Giá trị nhãn','Toxicity':'Mức độ độc hại','Constructiveness':'Tính xây dựng','Share (%)':'Tỷ lệ (%)',
    'Whitespace units':'Đơn vị khoảng trắng','Characters':'Ký tự','Occurrences':'Số lần xuất hiện','Distinct units':'Số đơn vị khác nhau','Mean TF-IDF':'TF-IDF trung bình',
    'Topic':'Chủ đề','Whitespace units per comment':'Đơn vị khoảng trắng / bình luận','Comments within class (%)':'Bình luận trong lớp (%)','Unique units / total units':'Số đơn vị khác nhau / tổng số đơn vị',
    'Segmented words per comment':'Số từ được tách / bình luận','Task':'Bài toán','Toxicity label':'Nhãn độc hại','Train':'Train','Validation':'Validation','Test':'Test',
    'Label 0':'Nhãn 0','Label 1':'Nhãn 1','Non-toxic':'Không độc hại','Toxic':'Độc hại','Non-toxic (0)':'Không độc hại (0)','Toxic (1)':'Độc hại (1)',
    'Non-constructive (0)':'Không xây dựng (0)','Constructive (1)':'Xây dựng (1)','Contains !':'Có dấu !','Contains ?':'Có dấu ?','Contains a digit':'Có chữ số','Contains a URL':'Có URL','Contains uppercase letters':'Có chữ hoa'
  };
  const trPlot = text => {
    if (typeof text !== 'string') return text;
    if (plotLabels[text]) return plotLabels[text];
    return text.replace('Comments:', 'Bình luận:').replace('Characters:', 'Ký tự:').replace('Whitespace units:', 'Đơn vị khoảng trắng:').replace(' words:', ' từ:').replace(' comments', ' bình luận').replace('Frequency —', 'Tần suất —').replace('Mean TF-IDF —', 'TF-IDF trung bình —');
  };
  function translatePlot(plot, index) {
    plot.layout.title.text = chartTitles[index] || plot.layout.title.text;
    for (const key of ['xaxis', 'yaxis']) {
      const axis = plot.layout[key];
      if (axis && axis.title && axis.title.text) axis.title.text = trPlot(axis.title.text);
      if (axis && axis.categoryarray) axis.categoryarray = axis.categoryarray.map(trPlot);
    }
    if (plot.layout.legend && plot.layout.legend.title) plot.layout.legend.title.text = trPlot(plot.layout.legend.title.text);
    plot.data.forEach(trace => {
      if (trace.name) trace.name = trPlot(trace.name);
      if (trace.hovertemplate) trace.hovertemplate = trPlot(trace.hovertemplate);
      if ([0, 1, 2, 9, 21].includes(index)) for (const axis of ['x', 'y']) if (Array.isArray(trace[axis])) trace[axis] = trace[axis].map(trPlot);
    });
    (plot.layout.updatemenus || []).forEach(menu => (menu.buttons || []).forEach(button => {
      button.label = trPlot(button.label);
      const update = button.args && button.args[1];
      if (update && typeof update.title === 'string') update.title = trPlot(update.title);
    }));
    return plot;
  }
  function applyCharts(lang) {
    if (!chartSource || !window.Plotly) return;
    chartSource.forEach((original, index) => {
      const el = document.getElementById(`plot-${index}`);
      if (!el) return;
      const plot = JSON.parse(JSON.stringify(original));
      if (index === 1 && plot.data[0]) plot.data[0].name = 'Toxicity';
      if (lang === 'vi') translatePlot(plot, index);
      Plotly.react(el, plot.data, plot.layout, {responsive: true, displaylogo: false, displayModeBar: false});
    });
  }
  fetch(dictionaryUrl).then(response => {
    if (!response.ok) throw new Error('Translation data unavailable');
    return response.json();
  }).then(data => {
    function apply(lang) {
      selected = lang;
      document.documentElement.lang = lang;
      document.title = data[lang][normalize(originalTitle)] || originalTitle;
      for (const entry of nodes) {
        const translated = data[lang][entry.normalized];
        entry.node.nodeValue = translated === undefined ? entry.raw : entry.raw.replace(entry.raw.trim(), translated);
      }
      for (const item of attributes) item.element.setAttribute(item.name, data[lang][item.original] || item.original);
      document.querySelectorAll('.chart[role=img], .text-eda-plot[role=img]').forEach(chart => { const label = chart.closest('figure')?.querySelector('.figure-label span'); if (label) chart.setAttribute('aria-label', label.textContent.trim()); });
      controls.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.language === lang)));
      document.querySelectorAll('.static-chart-note').forEach(note => { note.hidden = lang !== 'vi'; });
      const input = document.getElementById('word-count-input');
      const output = document.getElementById('word-count-result');
      if (input && output) {
        const n = Number(input.value);
        output.textContent = input.value !== '' && Number.isInteger(n) && n >= 0 && n < wordCounts.length ? `${wordCounts[n].toLocaleString(lang === 'vi' ? 'vi-VN' : 'en-US')} ${lang === 'vi' ? 'bình luận' : 'comments'}` : (lang === 'vi' ? `Nhập số nguyên từ 0 đến ${wordCounts.length - 1}.` : `Enter a whole number from 0 to ${wordCounts.length - 1}.`);
      }
      applyCharts(lang);
      try { localStorage.setItem(key, lang); } catch (_) { /* storage can be unavailable */ }
    }
    document.addEventListener('co5177:track-changed', () => {
      const panel = document.getElementById('track-panel');
      if (!panel) return;
      for (let i = nodes.length - 1; i >= 0; i--) if (!nodes[i].node.isConnected) nodes.splice(i, 1);
      const tw = document.createTreeWalker(panel, NodeFilter.SHOW_TEXT);
      while (tw.nextNode()) {
        const node = tw.currentNode;
        if (node.nodeValue.trim() && !nodes.some(entry => entry.node === node)) nodes.push({node, raw: node.nodeValue, normalized: normalize(node.nodeValue)});
      }
      apply(selected);
    });
    controls.forEach(button => button.addEventListener('click', () => apply(button.dataset.language)));
    const input = document.getElementById('word-count-input');
    if (input) input.addEventListener('input', () => { if (selected === 'vi') { const n = Number(input.value); const out = document.getElementById('word-count-result'); out.textContent = input.value !== '' && Number.isInteger(n) && n >= 0 && n < wordCounts.length ? `${wordCounts[n].toLocaleString('vi-VN')} bình luận` : `Nhập số nguyên từ 0 đến ${wordCounts.length - 1}.`; } });
    apply(selected);
  }).catch(() => {
    controls.forEach(button => { button.disabled = true; button.title = 'Language data could not be loaded'; });
  });
})();
