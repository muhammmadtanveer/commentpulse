document.getElementById("grab").addEventListener("click", async () => {
  const msg = document.getElementById("msg");
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.id) { msg.textContent = "No active tab found."; return; }
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: window.GRAB_FN });
    msg.textContent = "Done — check the page message and your downloads.";
  } catch (e) {
    msg.textContent = "Could not run on this page. Open a normal post page (not a chrome:// or settings page) and try again.";
  }
});
