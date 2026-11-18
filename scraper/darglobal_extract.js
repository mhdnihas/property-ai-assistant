/*
 * DarGlobal extractor — run in the browser DevTools console on https://darglobal.co.uk
 *
 * Why a browser and not requests/curl?
 *   darglobal.co.uk sits behind Imperva bot protection: plain HTTP clients get a
 *   212-byte challenge page. We do NOT try to bypass that. Instead, the pages are
 *   read from a normal browser session, at a polite pace (1 page / 1.5 s), and
 *   only from paths that robots.txt allows.
 *
 * How it works:
 *   The site is a Next.js app backed by a Strapi CMS. Every page ships its
 *   content as JSON in <script id="__NEXT_DATA__">. We walk that JSON and keep
 *   human-readable strings (titles, descriptions, unit types, amenities, FAQs),
 *   dropping image URLs, icon names, SEO blobs and duplicates.
 *
 * Output: downloads darglobal_pages.json -> move it to data/raw/
 */
(async () => {
  const PATHS = [
    "/", "/about", "/faq", "/why-invest", "/investor", "/hospitality", "/development-management",
    "/partners", "/one-of-one", "/projects", "/luxury-golf-communities", "/live-all-in",
    "/aida-360", "/aida-coastal-investment-villas", "/aida-trump-international-apartments",
    "/aida-trump-international-cliff-villas", "/aida-trump-international-hotel", "/aida-trump-villas",
    "/amaya/amaya-plots", "/amaya/padel-living", "/amour-sans-detour", "/amour-sans-detour-2",
    "/davinci-tower-by-pagani", "/dg1", "/d-villas-at-jge", "/fairway-villas-aida-oman", "/invest-in-aida",
    "/La-Collina", "/les-vagues", "/marea", "/marriott-residences-aida-oman", "/neptune",
    "/rayana/darglobal-mansions", "/rayana/darglobal-mansions-fendi-casa", "/rayana/trump-international-qatar",
    "/rayana/trump-mansions", "/Sunrise-Haven-Luxury-Villas", "/the-astera", "/the-great-escape-1-aida-oman",
    "/the-great-escape-2-aida-oman", "/the-mulliner", "/tierra-viva", "/trump-international-hotel-and-tower-dubai",
    "/trump-plaza/executive-residences", "/trump-plaza/park-residences", "/trump-tower-jeddah",
    "/urban-oasis-by-missoni", "/w-residences",
  ];
  const SKIP_KEY = /^(url|alternativeText|caption|mime|hash|ext|provider|provider_metadata|formats|icon|iconName|type|slug|vuid|createdAt|updatedAt|publishedAt|versionComment|locale|color|bgColor|className|link|href|videoUrl|metaTitle|metaDescription|keywords|canonicalURL|structuredData|metaRobots|metaViewport|ctaText|buttonText|email|phone|whatsapp|placeholder|label_ar)$/i;
  const SKIP_OBJ = new Set(["otherProjectData", "OtherProject", "globalData", "localizations", "thankyouPageData",
    "SEO", "seo", "PhotoGallery", "newGallery", "View360", "downloadbrochure", "bookunit", "versions"]);

  const extract = async (path) => {
    const res = await fetch(path, { credentials: "include" });
    const doc = new DOMParser().parseFromString(await res.text(), "text/html");
    const nd = doc.getElementById("__NEXT_DATA__");
    const base = { path, url: "https://darglobal.co.uk" + path, status: res.status, title: doc.title,
      desc: doc.querySelector("meta[name=description]")?.content || "" };
    if (!nd) return { ...base, error: "no __NEXT_DATA__" };
    const seen = new Set(), out = [];
    const walk = (o, k = "") => {
      if (o == null) return;
      if (typeof o === "string") {
        if (SKIP_KEY.test(k)) return;
        const s = o.replace(/<[^>]+>/g, " ").replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/\s+/g, " ").trim();
        if (s.length < 3 || /^https?:|^\/|^#/.test(s) || /\.(png|jpe?g|webp|svg|pdf|mp4|gif)$/i.test(s)
            || /^[a-z0-9]+(-[a-z0-9]+)*$/.test(s) || seen.has(s)) return;
        seen.add(s); out.push(s); return;
      }
      if (typeof o === "number" && /floor|area|price|bed|year/i.test(k)) { out.push(`${k}: ${o}`); return; }
      if (Array.isArray(o)) return o.forEach((x) => walk(x, k));
      if (typeof o === "object") for (const kk of Object.keys(o)) if (!SKIP_OBJ.has(kk)) walk(o[kk], kk);
    };
    walk(JSON.parse(nd.textContent).props.pageProps);
    return { ...base, text: out.join(" | ") };
  };

  const pages = [];
  for (const p of PATHS) {
    try { pages.push(await extract(p)); } catch (e) { pages.push({ path: p, error: String(e) }); }
    await new Promise((r) => setTimeout(r, 1500)); // be polite
  }
  const blob = new Blob([JSON.stringify({ scraped_at: new Date().toISOString(), source: "darglobal.co.uk", pages }, null, 1)],
    { type: "application/json" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "darglobal_pages.json" });
  document.body.appendChild(a); a.click(); a.remove();
  console.log(`done: ${pages.length} pages`);
})();
