export async function analyze(jd: string, file: File) {
  const form = new FormData();
  form.append("jd_text", jd);
  form.append("resume", file);
  const api = import.meta.env.VITE_API_URL || "http://localhost:8000";
  const res = await fetch(api + "/analyze", { method: "POST", body: form });
  if (!res.ok) throw new Error("Request failed");
  return res.json();
}
