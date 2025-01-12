"""PDF type implementations for AMICA."""

import numpy as np
from scipy import special

def compute_pdf(y: np.ndarray, rho: float, pdftype: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PDF value and derivative based on PDF type.
    
    Parameters
    ----------
    y : ndarray
        Activation values
    rho : float
        Shape parameter
    pdftype : int
        PDF type:
        1: Generalized Gaussian
        2: Logistic
        3: Generalized Logistic
        4: Gaussian Mixture
        
    Returns
    -------
    pdf : ndarray
        PDF values
    dpdf : ndarray
        PDF derivatives
    """
    if pdftype == 1:
        # Generalized Gaussian
        if rho == 1.0:
            # Laplace distribution
            pdf = np.exp(-np.abs(y)) / 2.0
            dpdf = -np.sign(y) * pdf
        elif rho == 2.0:
            # Gaussian distribution
            pdf = np.exp(-y*y) / np.sqrt(np.pi)
            dpdf = -2 * y * pdf
        else:
            # General case
            pdf = np.exp(-np.power(np.abs(y), rho)) / (
                2.0 * special.gammaln(1.0 + 1.0/rho))
            dpdf = -rho * np.power(np.abs(y), rho-1) * np.sign(y) * pdf
            
    elif pdftype == 2:
        # Logistic distribution
        exp_y = np.exp(y)
        pdf = exp_y / np.square(1 + exp_y)
        dpdf = pdf * (1 - exp_y) / (1 + exp_y)
        
    elif pdftype == 3:
        # Generalized logistic
        exp_y = np.exp(y)
        pdf = rho * exp_y / np.power(1 + exp_y, rho + 1)
        dpdf = pdf * (1 - (rho + 1)*exp_y/(1 + exp_y))
        
    elif pdftype == 4:
        # Gaussian mixture
        pdf1 = np.exp(-0.5*y*y) / np.sqrt(2*np.pi)
        pdf2 = np.exp(-0.5*y*y/rho) / np.sqrt(2*np.pi*rho)
        pdf = 0.5 * (pdf1 + pdf2)
        dpdf = -0.5 * (y*pdf1 + y*pdf2/rho)
        
    else:
        raise ValueError(f"Unknown PDF type: {pdftype}")
        
    return pdf, dpdf

def compute_log_pdf(y: np.ndarray, rho: float, pdftype: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute log PDF value and its derivative.
    
    Parameters
    ----------
    y : ndarray
        Activation values
    rho : float
        Shape parameter
    pdftype : int
        PDF type (see compute_pdf for details)
        
    Returns
    -------
    log_pdf : ndarray
        Log PDF values
    dlog_pdf : ndarray
        Log PDF derivatives
    """
    pdf, dpdf = compute_pdf(y, rho, pdftype)
    log_pdf = np.log(pdf)
    dlog_pdf = dpdf / pdf
    return log_pdf, dlog_pdf

def choose_pdf_type(data: np.ndarray, rho: float = 1.5) -> int:
    """
    Choose best PDF type based on data statistics.
    
    Parameters
    ----------
    data : ndarray
        Data samples
    rho : float
        Initial shape parameter
        
    Returns
    -------
    pdftype : int
        Selected PDF type
    """
    # Compute statistics
    kurt = np.mean(data**4) / np.square(np.mean(data**2)) - 3
    skew = np.mean(data**3) / np.power(np.mean(data**2), 1.5)
    
    if abs(kurt) < 0.5 and abs(skew) < 0.5:
        # Close to Gaussian
        return 1
    elif kurt > 2:
        # Heavy tailed
        return 2
    elif abs(skew) > 1:
        # Asymmetric
        return 3
    else:
        # Multimodal
        return 4
