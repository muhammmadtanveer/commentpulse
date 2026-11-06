/* CommentPulse Link Grabber — bookmarklet SOURCE (not served directly).
   The dashboard carries this as a `javascript:` bookmark the user drags to
   their bookmarks bar. It runs ONLY in the user's own browser, on the post
   page they are viewing: it reads the comments currently loaded on the page
   and downloads them as a CSV in CommentPulse import format
   (text,timestamp,platform,author). No password, no server scraping —
   nothing leaves the user's browser until they upload the file themselves.

   To rebuild the encoded href after editing: python3 tools/build_grabber.py
*/
(function () {
  var host = location.hostname;
  var plat = host.includes("facebook") ? "facebook"
    : host.includes("instagram") ? "instagram"
    : host.includes("tiktok") ? "tiktok"
    : host.includes("linkedin") ? "linkedin"
    : host.includes("pinterest") ? "pinterest" : "web";
  var rows = [], seen = {};
  function clean(s) { return (s || "").replace(/\s+/g, " ").trim(); }
  function textOf(el) { return el ? (el.innerText || el.textContent || "") : ""; }
  function add(text, author) {
    text = clean(text);
    if (text.length < 2 || text.length > 1200) return;
    var ui = ["Like", "Reply", "Share", "Follow", "Following", "See more",
      "View more comments", "Most relevant", "All comments",
      "Write a comment", "Add a comment", "Hide", "Report"];
    if (ui.indexOf(text) >= 0) return;
    var k = text.toLowerCase().slice(0, 140);
    if (seen[k]) return;
    seen[k] = 1;
    rows.push({ t: text, a: clean(author || "") });
  }
  try {
    if (plat === "facebook") {
      document.querySelectorAll('div[role="article"]').forEach(function (a) {
        var lab = a.getAttribute("aria-label") || "";
        if (!/comment/i.test(lab)) return;
        var author = clean(lab.replace(/^Comment by /i, "").split(" at ")[0]);
        var body = "";
        a.querySelectorAll('div[dir="auto"]').forEach(function (d) {
          var t = textOf(d);
          if (t && t.length > body.length && t.indexOf(author) !== 0) body = t;
        });
        if (!body) body = textOf(a).replace(author, "");
        add(body, author);
      });
    } else if (plat === "tiktok") {
      document.querySelectorAll('[data-e2e="comment-level-1"],[data-e2e="comment-level-2"]').forEach(function (c) {
        var u = c.querySelector('[data-e2e="comment-username-1"],[data-e2e="comment-username-2"]');
        var tx = c.querySelector("p");
        add(textOf(tx || c), textOf(u));
      });
    } else if (plat === "linkedin") {
      document.querySelectorAll(".comments-comment-item").forEach(function (c) {
        var b = c.querySelector(".comments-comment-item__main-content");
        var u = c.querySelector(".comments-post-meta__name-text,.comments-comment-meta__description-title");
        add(textOf(b), textOf(u));
      });
    } else if (plat === "instagram") {
      document.querySelectorAll("li").forEach(function (li) {
        var h = li.querySelector("h3,h2");
        if (!h) return;
        var best = "";
        li.querySelectorAll("span").forEach(function (s) {
          var t = textOf(s);
          if (t && t.length > best.length && t !== textOf(h)) best = t;
        });
        add(best, textOf(h));
      });
    }
    if (rows.length < 3) {
      var scope = document.querySelector('main,[role="main"],article') || document.body;
      scope.querySelectorAll("p,span,div").forEach(function (el) {
        if (el.children.length > 3) return;
        if (el.closest("header,nav,footer,aside")) return;
        add(textOf(el), "");
      });
    }
  } catch (e) { /* best effort — fall through to the count check */ }
  if (!rows.length) {
    alert("CommentPulse Grabber: no comments found on this page yet.\n\nScroll down to load comments, click \"View more comments\" to expand them, then click the Grabber again.");
    return;
  }
  function esc(v) {
    v = (v == null ? "" : String(v));
    return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  }
  var csv = "text,timestamp,platform,author\n" + rows.map(function (r) {
    return [esc(r.t), "", plat, esc(r.a)].join(",");
  }).join("\n");
  var blob = new Blob([csv], { type: "text/csv" });
  var a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "commentpulse-comments.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  alert("CommentPulse Grabber: " + rows.length + " comments saved to commentpulse-comments.csv\n\nNow go back to CommentPulse and upload that file in the \"Upload CSV\" tab — your analysis runs instantly.");
})()
