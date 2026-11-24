"""The image-field library in manuscript equation (3), in publication order."""
TERMS = ("1", "u", "u2", "ux", "uy", "uxx", "uyy", "uxy",
         "u_ux", "u_uy", "u_uxx", "u_uyy", "u_uxy")


def library(u, ux, uy, uxx, uyy, uxy):
    return (u * 0 + 1, u, u * u, ux, uy, uxx, uyy, uxy,
            u * ux, u * uy, u * uxx, u * uyy, u * uxy)
