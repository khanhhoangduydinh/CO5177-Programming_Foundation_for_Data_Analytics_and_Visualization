const trackData = {
  tabular: {
    tabId: "tab-tabular",
    type: "CORE TRACK · BẮT BUỘC",
    title: "Tabular Data",
    description:
      "Khám phá các mối quan hệ ẩn trong dữ liệu dạng bảng bằng thống kê, trực quan và machine learning có thể giải thích.",
    items: [
      "≥ 2.000 samples, ≥ 10 cột",
      "Missing values & outliers",
      "Categorical + numerical features",
    ],
    name: "dataset.csv",
    count: "2.000+ rows",
    preview: [
      '<div class="table-head"><span>id</span><span>category</span><span>value</span><span>target</span></div>',
      '<div><span>#0842</span><span class="pill pill-yellow">alpha</span><span>18.42</span><span>1</span></div>',
      '<div><span>#0843</span><span class="pill pill-mint">beta</span><span>—</span><span>0</span></div>',
      '<div><span>#0844</span><span class="pill pill-orange">gamma</span><span>31.07</span><span>1</span></div>',
      '<div><span>#0845</span><span class="pill pill-yellow">alpha</span><span>22.91</span><span>0</span></div>',
      '<div><span>#0846</span><span class="pill pill-mint">beta</span><span>14.66</span><span>1</span></div>',
    ].join(""),
    previewClass: "preview-table",
  },
  text: {
    tabId: "tab-text",
    type: "CORE TRACK · BẮT BUỘC VỚI NHÓM",
    title: "Text Data",
    description:
      "Biến ngôn ngữ tự nhiên thành tín hiệu định lượng qua làm sạch văn bản, biểu diễn đặc trưng và phân loại.",
    items: [
      "≥ 2.000 văn bản",
      "Ưu tiên dữ liệu tiếng Việt",
      "Crawling dữ liệu riêng được khuyến khích",
    ],
    name: "corpus_vi.json",
    count: "2.000+ documents",
    preview: [
      '<div class="text-bubble">“Trải nghiệm học tập rất <mark>trực quan</mark> và dễ theo dõi.”</div>',
      '<div class="text-bubble">label: <strong>positive</strong> · confidence: 0.94</div>',
      '<div class="text-bubble">“Mô hình cần xử lý tốt hơn với câu có <mark>phủ định</mark>.”</div>',
      '<div class="text-bubble">tokens: 14 · language: vi · cleaned: true</div>',
    ].join(""),
    previewClass: "preview-text",
  },
  image: {
    tabId: "tab-image",
    type: "OPTIONAL TRACK · MỞ RỘNG",
    title: "Image Data",
    description:
      "Kết hợp OpenCV và pretrained model để đọc cấu trúc thị giác, trích đặc trưng và phân loại ảnh.",
    items: [
      "≥ 5.000 ảnh, ≥ 3 lớp",
      "Cạnh ngắn khuyến nghị ≥ 128 px",
      "Split rõ ràng, augmentation hợp lý",
    ],
    name: "images/train/",
    count: "5.000+ images",
    preview: [
      '<div class="pixel-card"><span>class_01 · .97</span></div>',
      '<div class="pixel-card"><span>class_02 · .91</span></div>',
      '<div class="pixel-card"><span>class_03 · .89</span></div>',
      '<div class="pixel-card"><span>class_01 · .94</span></div>',
      '<div class="pixel-card"><span>class_03 · .93</span></div>',
      '<div class="pixel-card"><span>class_02 · .88</span></div>',
    ].join(""),
    previewClass: "preview-image-grid",
  },
};

const header = document.querySelector("[data-header]");
const menuToggle = document.querySelector("[data-menu-toggle]");
const nav = document.querySelector("[data-nav]");
const navLinks = [...document.querySelectorAll(".site-nav a")];
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function updateHeader() {
  header?.classList.toggle("is-scrolled", window.scrollY > 18);
}

updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

menuToggle?.addEventListener("click", () => {
  const isOpen = menuToggle.getAttribute("aria-expanded") === "true";
  menuToggle.setAttribute("aria-expanded", String(!isOpen));
  menuToggle.setAttribute("aria-label", isOpen ? "Mở menu" : "Đóng menu");
  nav.classList.toggle("is-open", !isOpen);
  document.body.style.overflow = isOpen ? "" : "hidden";
});

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    menuToggle?.setAttribute("aria-expanded", "false");
    menuToggle?.setAttribute("aria-label", "Mở menu");
    nav?.classList.remove("is-open");
    document.body.style.overflow = "";
  });
});

const revealElements = document.querySelectorAll(".reveal");
if (reducedMotion || !("IntersectionObserver" in window)) {
  revealElements.forEach((element) => element.classList.add("is-visible"));
} else {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -35px" },
  );
  revealElements.forEach((element) => revealObserver.observe(element));
}

const sections = document.querySelectorAll("main section[id]");
if ("IntersectionObserver" in window) {
  const sectionObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        navLinks.forEach((link) => {
          link.classList.toggle(
            "is-active",
            link.getAttribute("href") === "#" + entry.target.id,
          );
        });
      });
    },
    { rootMargin: "-35% 0px -60% 0px" },
  );
  sections.forEach((section) => sectionObserver.observe(section));
}

const trackButtons = [...document.querySelectorAll("[data-track]")];
const trackPanel = document.querySelector("#track-panel");
const trackCopy = document.querySelector(".track-copy");
const trackPreview = document.querySelector(".data-preview");
const trackType = document.querySelector("[data-track-type]");
const trackTitle = document.querySelector("[data-track-title]");
const trackDescription = document.querySelector("[data-track-description]");
const trackList = document.querySelector("[data-track-list]");
const previewName = document.querySelector("[data-preview-name]");
const previewCount = document.querySelector("[data-preview-count]");
const previewVisual = document.querySelector("[data-preview]");

function switchTrack(trackName) {
  const next = trackData[trackName];
  if (!next) return;

  trackButtons.forEach((button) => {
    const selected = button.dataset.track === trackName;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });

  trackCopy.classList.add("is-changing");
  trackPreview.classList.add("is-changing");

  window.setTimeout(() => {
    trackType.textContent = next.type;
    trackTitle.textContent = next.title;
    trackDescription.textContent = next.description;
    trackList.innerHTML = next.items.map((item) => "<li>" + item + "</li>").join("");
    previewName.textContent = next.name;
    previewCount.textContent = next.count;
    previewVisual.className = "preview-visual " + next.previewClass;
    previewVisual.innerHTML = next.preview;
    trackPanel.setAttribute("aria-labelledby", next.tabId);

    trackCopy.classList.remove("is-changing");
    trackPreview.classList.remove("is-changing");
  }, reducedMotion ? 0 : 155);
}

trackButtons.forEach((button, index) => {
  button.addEventListener("click", () => switchTrack(button.dataset.track));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    const targetIndex = (index + direction + trackButtons.length) % trackButtons.length;
    trackButtons[targetIndex].focus();
    switchTrack(trackButtons[targetIndex].dataset.track);
  });
});

const parallaxCard = document.querySelector("[data-parallax]");
if (parallaxCard && !reducedMotion && window.matchMedia("(pointer: fine)").matches) {
  parallaxCard.addEventListener("pointermove", (event) => {
    const bounds = parallaxCard.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width - 0.5;
    const y = (event.clientY - bounds.top) / bounds.height - 0.5;
    parallaxCard.style.transform =
      "perspective(1000px) rotateX(" + -y * 2.2 + "deg) rotateY(" + x * 2.2 + "deg)";
  });
  parallaxCard.addEventListener("pointerleave", () => {
    parallaxCard.style.transform = "";
  });
}
