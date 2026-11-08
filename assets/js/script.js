const trackData = {
  tabular: {
    tabId: "tab-tabular",
    type: "CORE TRACK · BẮT BUỘC",
    title: "Tabular Data",
    description:
      "Khám phá các mối quan hệ ẩn trong dữ liệu dạng bảng bằng thống kê, trực quan và machine learning có thể giải thích.",
    items: [
      "NASA Kepler KOI · 9.564 objects",
      "Leakage-safe EDA & classification",
      "PDF, notebook and report page",
    ],
    name: "Dữ liệu mẫu · Kepler KOI",
    count: "9.564 objects",
    reportLink: "tabular-eda.html",
    preview: [
      '<div class="table-head"><span>KOI</span><span>Phân loại</span><span>Chu kỳ (ngày)</span><span>Bán kính (R⊕)</span></div>',
      '<div><span>K00752.01</span><span class="pill pill-mint">CONFIRMED</span><span>9.48803557</span><span>2.26</span></div>',
      '<div><span>K00752.02</span><span class="pill pill-mint">CONFIRMED</span><span>54.4183827</span><span>2.83</span></div>',
      '<div><span>K00753.01</span><span class="pill pill-yellow">CANDIDATE</span><span>19.89913995</span><span>14.6</span></div>',
      '<div><span>K00754.01</span><span class="pill pill-orange">FALSE POSITIVE</span><span>1.736952453</span><span>33.46</span></div>',
      '<div><span>K00755.01</span><span class="pill pill-mint">CONFIRMED</span><span>2.525591777</span><span>2.75</span></div>',
    ].join(""),
    previewClass: "preview-table",
  },
  text: {
    tabId: "tab-text",
    type: "CORE TRACK · BẮT BUỘC VỚI NHÓM",
    title: "Text Data",
    description:
      "Phân tích 10.000 bình luận tiếng Việt của ViCTSD với kiểm tra rò rỉ dữ liệu, TF-IDF và phân loại nhị phân có thể giải thích.",
    items: [
      "ViCTSD · 10.000 Vietnamese comments",
      "Toxicity + Constructiveness",
      "EDA, PDF và Colab notebook",
    ],
    name: "Bình luận mẫu · ViCTSD",
    count: "10.000 comments",
    reportLink: "text-eda.html",
    preview: [
      '<div class="text-bubble">“Thật tuyệt vời...!!!”<small>Nhãn gốc: Toxicity 0 · Constructiveness 0</small></div>',
      '<div class="text-bubble">“Tại sao không lấy hạt quả táo của chính cây táo đó gieo trồng”<small>Nhãn gốc: Toxicity 0 · Constructiveness 1</small></div>',
      '<div class="text-bubble">“Ước mơ của mình là được ngắm cực quang một lần trong đời.”<small>Nhãn gốc: Toxicity 0 · Constructiveness 0</small></div>',
    ].join(""),
    previewClass: "preview-text",
  },
  image: {
    tabId: "tab-image",
    type: "OPTIONAL TRACK · MỞ RỘNG",
    title: "Intel Image Classification",
    description:
      "Phân loại sáu loại cảnh quan bằng EDA độc lập, augmentation hợp lý và MobileNetV3 pretrained dễ chạy lại.",
    items: [
      "17.034 ảnh RGB · 150 × 150 pixels",
      "6 lớp: buildings đến street",
      "Stratified split + test folder riêng",
    ],
    name: "Ảnh mẫu · Intel Image Classification",
    count: "17.034 labelled images",
    reportLink: "image-eda.html",
    preview: [
      '<div class="pixel-card"><img src="assets/intel-forest.jpg" alt="Ảnh rừng trong lớp forest"><span>forest</span></div>',
      '<div class="pixel-card"><img src="assets/intel-glacier.jpg" alt="Ảnh sông băng trong lớp glacier"><span>glacier</span></div>',
      '<div class="pixel-card"><img src="assets/intel-sea.jpg" alt="Ảnh biển trong lớp sea"><span>sea</span></div>',
      '<div class="pixel-card"><img src="assets/intel-street.jpg" alt="Ảnh đường phố trong lớp street"><span>street</span></div>',
      '<div class="pixel-card"><img src="assets/intel-mountain-example.jpg" alt="Ảnh núi trong lớp mountain"><span>mountain</span></div>',
      '<div class="pixel-card"><img src="assets/intel-buildings.jpg" alt="Ảnh tòa nhà trong lớp buildings"><span>buildings</span></div>',
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
const trackReportLink = document.querySelector("[data-report-link]");
const previewName = document.querySelector("[data-preview-name]");
const previewCount = document.querySelector("[data-preview-count]");
const previewVisual = document.querySelector("[data-preview]");

function switchTrack(trackName) {
  const next = trackData[trackName];
  if (!next) return;

  if (trackReportLink) {
    trackReportLink.hidden = !next.reportLink;
    if (next.reportLink) trackReportLink.href = next.reportLink;
  }

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
