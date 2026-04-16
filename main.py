import numpy as np
import cv2
from numba import njit, prange, config
import os
import time
import sys

os.environ["NUMBA_NUM_THREADS"] = "16"
config.THREADING_LAYER = "omp"


@njit(parallel=True, fastmath=True, cache=True)
def compute_full_frame(
    RES, STEPS, DT, RS, t, x, y, cx, cy, cz, fx, fy, fz, rx, ry, rz, ux, uy, uz
):
    img = np.zeros((RES, RES, 3), dtype=np.float32)
    dist_to_hole = np.sqrt(cx * cx + cy * cy + cz * cz)

    for i in prange(RES):
        for j in range(RES):
            vx = fx + x[0, j] * rx * 0.25 + y[i, 0] * ux * 0.25
            vy = fy + x[0, j] * ry * 0.25 + y[i, 0] * uy * 0.25
            vz = fz + x[0, j] * rz * 0.25 + y[i, 0] * uz * 0.25
            vn = np.sqrt(vx * vx + vy * vy + vz * vz)
            vx /= vn
            vy /= vn
            vz /= vn

            px, py, pz = cx, cy, cz
            inst = 0.0
            min_r = 100.0
            acc_light = 0.0
            hit_hole = False

            for s in range(STEPS):
                r2 = px * px + py * py + pz * pz
                r = np.sqrt(r2)
                if r < min_r:
                    min_r = r

                if r < RS * 1.005 and acc_light < 0.7:
                    hit_hole = True
                    break

                py_next = py + vy * DT
                if py * py_next <= 0:
                    th = -py / (vy + 1e-9)
                    hx, hz = px + vx * th, pz + vz * th
                    r2_d = hx * hx + hz * hz
                    if 3.5 < r2_d < 100.0:
                        rv = np.sqrt(r2_d)
                        ang = np.arctan2(hz, hx)
                        wave = (
                            np.sin(0.8 * rv - ang * 1.5 + t * 5.0)
                            + 0.4 * np.sin(2.5 * rv + ang * 3.0)
                        ) / 1.4

                        radial_boost = 120.0 / (rv**5 + 1.0)

                        dop = 1.0 + 0.6 * np.sin(ang)
                        # val = (1 + wave) * dop * radial_boost * np.exp(-0.05 * rv) * 2.3
                        val = (1 + wave) * dop * radial_boost * np.exp(-0.02 * rv) * 3.2

                        dist_hit = np.sqrt(
                            (hx - cx) ** 2 + (py - cy) ** 2 + (hz - cz) ** 2
                        )
                        dist_to_hole = np.sqrt(cx**2 + cy**2 + cz**2)

                        if dist_hit < dist_to_hole:
                            inst += val * 2
                            acc_light += 0.75
                            if acc_light >= 1.0:
                                break
                        else:
                            val *= 0.3
                            inst += val * 0.43

                a = -0.95 * RS / (r2 * r + 1e-6)

                vx += px * a * DT
                vy += py * a * DT
                vz += pz * a * DT
                px += vx * DT
                py += vy * DT
                pz += vz * DT
                if r > 120.0:
                    break

            if hit_hole:
                img[i, j] = 0.0
            else:
                ps = np.exp(-2000 * (min_r - 1.01) ** 2) * 9.0

                glow = np.exp(-1.2 * (min_r - 1.2) ** 2) * 0.6

                r_c = inst * 5.0 + ps * 3.0 + glow * 1.5
                g = inst * 3.5 + ps * 3.0 + glow * 0.8
                b = inst * 2.2 + ps * 3.0 + glow * 0.4

                img[i, j, 0] = 1.0 - np.exp(-b)
                img[i, j, 1] = 1.0 - np.exp(-g)
                img[i, j, 2] = 1.0 - np.exp(-r_c)
    return img


def render():
    RES = 300  # 500
    STEPS = 700
    DT = 0.12

    y, x = np.ogrid[1 : -1 : complex(RES), -1 : 1 : complex(RES)]

    sys.stdout.flush()

    frame_idx = 0

    cv2.namedWindow("Black Hole")
    os.system("cls" if os.name == "nt" else "clear")
    print("Запуск рендера...")
    while True:
        start_t = time.time()
        t = (frame_idx / 300) * 2 * np.pi

        dist = 45.0
        vertical_angle = -np.radians(15) * np.sin(t * 0.2)

        horizontal_dist = dist * np.cos(vertical_angle)
        cx, cy, cz = (
            np.cos(t) * horizontal_dist,
            dist * np.sin(vertical_angle),
            np.sin(t) * horizontal_dist,
        )

        # cx, cy, cz = np.cos(t) * dist, np.sin(t * 0.2) * -15.0, np.sin(t) * dist

        cp = np.array([cx, cy, cz])
        fw = -cp / np.linalg.norm(cp)

        rt_base = np.array([-np.sin(t), 0, np.cos(t)])

        up_base = np.cross(rt_base, fw)
        up_base /= np.linalg.norm(up_base)

        tilt_angle = (np.pi / 12) * np.sin(t * 0.3)

        rt = rt_base * np.cos(tilt_angle) + np.cross(fw, rt_base) * np.sin(tilt_angle)
        up = up_base * np.cos(tilt_angle) + np.cross(fw, up_base) * np.sin(tilt_angle)

        frame = compute_full_frame(
            RES,
            STEPS,
            DT,
            1.0,
            t,
            x,
            y,
            cx,
            cy,
            cz,
            fw[0],
            fw[1],
            fw[2],
            rt[0],
            rt[1],
            rt[2],
            up[0],
            up[1],
            up[2],
        )

        cv2.imshow("Black Hole", frame)

        curr_fps = 1.0 / (time.time() - start_t)
        if frame_idx % 10 == 0:
            print(f"FPS: {curr_fps:.2f} | Frame: {frame_idx}", end="\r")

        frame_idx += 1
        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":

    try:
        render()
    except Exception as e:
        print(f"\nОШИБКА ПРИ ЗАПУСКЕ: {e}")
