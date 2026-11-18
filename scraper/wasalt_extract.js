/*
 * Wasalt extractor — run in the browser DevTools console on https://wasalt.sa/en
 *
 * Why a browser? wasalt.sa is behind a Cloudflare challenge for non-browser
 * clients. We do not bypass it; we read public category pages from a normal
 * browser session at 1 page / 2 s. robots.txt allows these pages (only /search
 * is disallowed for generic bots).
 *
 * Each category page (e.g. /en/villas-for-sale-in-riyadh) is a Next.js page whose
 * __NEXT_DATA__ contains the first 32 listings with full details, the total
 * listing count, an FAQ and a market overview article. We keep a compact record
 * per listing.
 *
 * Output: downloads wasalt_pages.json -> move it to data/raw/
 * The listing *index* (all ~60k sale URLs) comes from the public sitemaps instead,
 * see scraper/wasalt_sitemap.py.
 */
(async () => {
  const CITIES = ["riyadh", "jeddah", "dammam", "al-khobar", "makkah", "madinah"];
  const TYPES = ["villas-for-sale", "apartments-for-sale", "lands-for-sale", "apartments-for-rent", "villas-for-rent"];
  const PAGES = CITIES.flatMap((c) => TYPES.map((t) => `/en/${t}-in-${c}`)).concat([
    "/en/buildings-for-sale-in-riyadh", "/en/floors-for-sale-in-riyadh",
    "/en/buildings-for-sale-in-jeddah", "/en/floors-for-sale-in-jeddah",
  ]);
  const strip = (s) => (s || "").replace(/<[^>]+>/g, " ").replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/\s+/g, " ").trim();

  const out = [];
  for (const path of PAGES) {
    try {
      const res = await fetch(path, { credentials: "include" });
      const doc = new DOMParser().parseFromString(await res.text(), "text/html");
      const nd = doc.getElementById("__NEXT_DATA__");
      if (!nd) { out.push({ path, status: res.status, error: "no __NEXT_DATA__" }); continue; }
      const pp = JSON.parse(nd.textContent).props.pageProps;
      const sr = pp.searchResult || {};
      const listings = (sr.properties || []).map((p) => {
        const i = p.propertyInfo || {};
        const attrs = {};
        (p.attributes || []).forEach((a) => (attrs[a.key] = a.unit ? `${a.value} ${a.unit}` : a.value));
        return {
          id: p.id, title: i.title,
          url: i.slug ? `https://wasalt.sa/en/property/${i.propertyFor}/${i.slug}` : null,
          purpose: i.propertyFor, type: i.propertySubType, main_type: i.propertyMainType,
          city: i.city, district: i.district, territory: i.territory, address: i.address,
          price: i.salePrice || i.conversionPrice || null, expected_rent: i.expectedRent || null,
          rent_frequency: i.rentFreq || null, currency: i.currencyType, furnishing: i.furnishingType,
          project: i.projectName, attributes: attrs,
          amenities: (p.featureAmenities || []).map((a) => a.name || a.title).filter(Boolean),
          lat: p.location?.lat, lon: p.location?.lon, published_at: p.publishedAt,
          verified: p.isVerified, rega_licensed: p.isRegaProp,
        };
      });
      out.push({
        path, url: "https://wasalt.sa" + path, status: res.status, title: pp.metaData?.h1tag || doc.title,
        total_count: sr.count,
        faq: (pp.FAQSchema?.mainEntity || []).map((q) => ({ q: q.name, a: strip(q.acceptedAnswer?.text) })),
        overview: strip(pp.para).slice(0, 6000), listings,
      });
    } catch (e) { out.push({ path, error: String(e) }); }
    await new Promise((r) => setTimeout(r, 2000)); // be polite
  }
  const blob = new Blob([JSON.stringify({ scraped_at: new Date().toISOString(), source: "wasalt.sa", pages: out }, null, 1)],
    { type: "application/json" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "wasalt_pages.json" });
  document.body.appendChild(a); a.click(); a.remove();
  console.log(`done: ${out.length} pages`);
})();
