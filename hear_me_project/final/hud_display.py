"""
Real-time OpenCV HUD overlay: confidence meter, current-sign label,
sentence bar, and FPS counter, drawn directly onto the camera feed.
"""
import time
import cv2
import config as cfg


class HUDRenderer:
    def __init__(self, width=cfg.FRAME_WIDTH, height=cfg.FRAME_HEIGHT):
        self.width = width
        self.height = height
        self._prev_time = time.time()
        self._fps = 0.0

    def _update_fps(self):
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = self._fps * 0.9 + inst * 0.1 if self._fps else inst
        return self._fps

    def _panel(self, frame, x, y, w, h, alpha=cfg.HUD_BG_ALPHA):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def draw(self, frame, current_label, confidence, vote_strength,
              sentence_text, last_confirmed):
        h, w = frame.shape[:2]
        fps = self._update_fps()

        # ---- top panel: current sign + confidence meter ----
        self._panel(frame, 0, 0, w, 90)
        cv2.putText(frame, f"Sign: {current_label}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.HUD_FONT_SCALE, cfg.COLOR_TEXT, 2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 130, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cfg.COLOR_TEXT, 1)

        bar_x, bar_y, bar_w, bar_h = 20, 55, w - 40, 18
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), cfg.COLOR_BAR_BG, -1)
        fill_w = int(bar_w * max(0.0, min(1.0, confidence)))
        bar_color = cfg.COLOR_CONFIRM if confidence >= cfg.MIN_CONFIDENCE else cfg.COLOR_LOW_CONF
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.putText(frame, f"{confidence * 100:.0f}%", (bar_x + bar_w - 45, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cfg.COLOR_TEXT, 1)

        # stabilization / vote-strength tick mark under the bar
        tick_x = bar_x + int(bar_w * vote_strength)
        cv2.line(frame, (tick_x, bar_y - 4), (tick_x, bar_y + bar_h + 4), cfg.COLOR_ACCENT, 2)

        # ---- bottom panel: sentence bar ----
        panel_h = 70
        self._panel(frame, 0, h - panel_h, w, panel_h)
        display_text = sentence_text if sentence_text else "(sentence will appear here)"
        cv2.putText(frame, display_text, (20, h - panel_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, cfg.COLOR_TEXT, 2)
        if last_confirmed:
            cv2.putText(frame, f"last confirmed: {last_confirmed}", (20, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cfg.COLOR_ACCENT, 1)

        # ---- key hints ----
        cv2.putText(frame, "[SPACE]=speak sentence  [c]=clear  [q]=quit",
                    (20, h - panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1)

        return frame
