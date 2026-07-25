(function () {
  const measurementId = "G-P0QMF67972";
  const existingTag = document.querySelector(`script[src*="googletagmanager.com/gtag/js?id=${measurementId}"]`);

  window.dataLayer = window.dataLayer || [];
  window.gtag = window.gtag || function () {
    window.dataLayer.push(arguments);
  };

  if (!existingTag) {
    const script = document.createElement("script");
    script.async = true;
    script.src = `https://www.googletagmanager.com/gtag/js?id=${measurementId}`;
    document.head.appendChild(script);
    window.gtag("js", new Date());
    window.gtag("config", measurementId);
  }

  document.addEventListener("click", function (event) {
    const link = event.target.closest("[data-track]");
    if (!link || typeof window.gtag !== "function") return;

    window.gtag("event", link.dataset.track, {
      event_category: "outbound_cta",
      link_url: link.href,
      page_location: window.location.href
    });
  });
})();
