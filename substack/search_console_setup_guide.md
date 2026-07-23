# Getting tthl.substack.com Verified in Google Search Console

## Why this isn't a one-click thing

Substack doesn't give you a generic "paste this meta tag" field for site verification — the only built-in verification field under Settings > Analytics is labeled "Facebook Site Verification," and it's Facebook-specific. That rules out the HTML meta tag and HTML file upload methods Search Console normally offers.

That leaves two paths that actually work on Substack: connecting a custom domain (which unlocks DNS verification) or linking a Google Analytics or Google Tag Manager property (which unlocks the "Google Analytics"/"Google Tag Manager" verification methods in Search Console). Pick whichever fits — Path B (Analytics) is faster if you don't already own a domain; Path A (custom domain) is worth doing anyway if you want tthl.substack.com to eventually live at your own URL.

Once either path is verified, the last step is the same for both: submitting your RSS feed as the discovery mechanism, since Substack doesn't generate a sitemap.xml.

---

## Path A: Connect a custom domain, verify via DNS

1. **Buy a domain** if you don't have one (Namecheap, Google Domains/Squarespace, Cloudflare — any registrar works).
2. In Substack, go to **Settings > Domain**. You'll see "Add a custom domain — Set up your Substack to live on a domain you already own." Click **Add** and follow Substack's prompts; it will give you DNS records (usually a CNAME) to add at your registrar.
3. Add those records at your registrar and wait for Substack to confirm the domain is connected (can take anywhere from a few minutes to a few hours for DNS to propagate).
4. Go to [Google Search Console](https://search.google.com/search-console) and click **Add property**. Choose the **Domain** property type (not URL-prefix) and enter your bare domain (e.g. `yourdomain.com`).
5. Search Console will give you a **TXT record** to add. Go back to your domain registrar's DNS settings and add that TXT record.
6. Return to Search Console and click **Verify**. Domain-property verification checks DNS directly, so it doesn't matter that the record isn't related to Substack's own DNS setup.

---

## Path B: Link Google Analytics or Tag Manager, verify without a domain

This is the quicker route if you're not ready to buy a domain.

1. If you don't already have one, create a **GA4 property** at [analytics.google.com](https://analytics.google.com) for tthl.substack.com. Once created, copy its **Measurement ID** (format `G-XXXXXXXXXX`, found under Admin > Data Streams > your stream).
2. In Substack, go to **Settings > Analytics**. Paste the Measurement ID into the **"Google Analytics Measurement ID"** field and save. This makes Substack fire GA tracking on every page view.
3. Give it a little time to register traffic (an hour or so, or just visit your own site once GA is wired up), then go to [Google Search Console](https://search.google.com/search-console) and click **Add property**. Use the **URL-prefix** option and enter `https://tthl.substack.com`.
4. In the verification methods list, choose **Google Analytics**. Search Console will detect the GA property automatically if it's tracking the same site and you're logged in with the same Google account that administers that GA property. Click **Verify**.

If you'd rather use Tag Manager instead: create a container at [tagmanager.google.com](https://tagmanager.google.com), copy the **Container ID** (format `GTM-XXXXXXX`), paste it into the **"Google Tag Manager ID"** field under Settings > Analytics, then in Search Console choose the **Google Tag Manager** verification method instead of Google Analytics.

---

## Final step for either path: submit the feed as your sitemap

Substack doesn't expose a sitemap.xml, so the RSS feed is the closest substitute for telling Google what to crawl.

1. In Search Console, with your property verified, open the **Sitemaps** section in the left sidebar.
2. Under "Add a new sitemap," enter `feed` (Search Console will append it to your property URL) — or the full path, `https://tthl.substack.com/feed`, if entering the full URL.
3. Click **Submit**. Google will periodically re-check the feed for new posts; it typically picks up newly published posts within a day or two once this is in place.

That's it — from here, Search Console's Coverage and Performance reports should start populating within a few days as Google crawls and indexes the site.
