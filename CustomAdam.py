import torch
from torch.optim import Optimizer


# Niestandardowy optymalizator Adam - rozszerzenie klasy Optimizer
class CustomAdam(Optimizer):
    """
    Niestandardowa implementacja optymalizatora Adam. Domyślne wartości są zgodne z zaleceniami zawartymi w https://arxiv.org/abs/1412.6980.
    Zobacz publikację lub plik Optimizer_Experimentation.ipynb, aby dowiedzieć się więcej o działaniu Adama oraz jego matematycznych podstawach.

    Parametry:
    stepsize (float): efektywna górna granica kroku optymalizatora (rozmiar kroku). DOMYŚLNIE - 0.001.
    bias_m1 (float): współczynnik dla pierwszego momentu. DOMYŚLNIE - 0.9
    bias_m2 (float): współczynnik dla drugiego, niecentrowanego momentu. DOMYŚLNIE - 0.999.
    epsilon (float): mała wartość dodawana w celu zapobieżenia dzieleniu przez zero. DOMYŚLNIE - 10e-8.
    bias_correction (bool): określa, czy optymalizator powinien korygować określone biasy podczas wykonywania kroku. DOMYŚLNIE - TRUE.
    """

    # Inicjalizacja optymalizatora z podanymi parametrami
    def __init__(self, params, stepsize=0.001, bias_m1=0.9, bias_m2=0.999, epsilon=10e-8, bias_correction=True):
        # Sprawdzenie, czy współczynnik kroku oraz wartości biasów są prawidłowe (czy nie są ujemne)
        if stepsize < 0:
            raise ValueError("Nieprawidłowy współczynnik kroku [{}]. Wybierz dodatnią wartość.".format(stepsize))
        if bias_m1 < 0 or bias_m2 < 0 and bias_correction:
            raise ValueError(
                "Nieprawidłowe parametry biasu [{}, {}]. Wybierz dodatnie wartości.".format(bias_m1, bias_m2))
        # Deklaracja słownika domyślnych wartości dla optymalizatora
        DEFAULTS = dict(stepsize=stepsize, bias_m1=bias_m1, bias_m2=bias_m2, epsilon=epsilon,
                        bias_correction=bias_correction)
        # Inicjalizacja optymalizatora
        super(CustomAdam, self).__init__(params, DEFAULTS)

    # Metoda wykonująca krok optymalizacji (aktualizację parametrów)
    def step(self, closure=None):
        # Ustawienie wartości straty na None
        loss = None
        # Jeśli podano closure, przypisanie wartości straty do wyniku closure
        loss = closure() if closure != None else loss
        # Sprawdzenie, czy jest to pierwszy krok - jeśli tak, ustawienie wartości na 1, w przeciwnym razie inkrementacja
        if not self.state["step"]:
            self.state["step"] = 1
        else:
            self.state["step"] += 1
        # Iteracja po grupach parametrów (warstwach sieci neuronowej) w celu rozpoczęcia obliczeń i aktualizacji parametrów
        for param_group in self.param_groups:
            # Iteracja po poszczególnych parametrach
            for param in param_group["params"]:
                # Sprawdzenie, czy dla danego parametru obliczono gradient
                # Jeśli gradienty nie są dostępne - pomijamy ten parametr
                if param.grad.data == None:
                    continue
                else:
                    gradients = param.grad.data
                # Zastosowanie metody Adam - na pierwszym kroku inicjalizacja wymaganych wartości dla danego parametru
                if self.state["step"] == 1:
                    # Ustawienie pierwszego i drugiego momentu na wektory zerowe
                    self.state["first_moment_estimate"] = torch.zeros_like(param.data)
                    self.state["second_moment_estimate"] = torch.zeros_like(param.data)
                # Deklaracja zmiennych na podstawie wartości w stanie optymalizatora (modyfikacje są wykonywane in-place)
                first_moment_estimate = self.state["first_moment_estimate"]
                second_moment_estimate = self.state["second_moment_estimate"]
                # Obliczenie pierwszego momentu - B_1 * m_t + (1-B_1) * grad (niecentrowany)
                first_moment_estimate.mul_(param_group["bias_m1"]).add_(gradients * (1.0 - param_group["bias_m1"]))
                # Obliczenie drugiego momentu - B_2 * v_t + (1-B_2) * grad^2 (niecentrowany)
                second_moment_estimate.mul_(param_group["bias_m2"]).add_(
                    gradients.pow_(2) * (1.0 - param_group["bias_m2"]))
                # Wykonanie korekty biasu, jeśli parametr jest ustawiony na True
                if param_group["bias_correction"]:
                    # Korekta biasu dla pierwszego momentu: m_t / (1 - B_1^t)
                    first_moment_estimate.divide_(1.0 - (param_group["bias_m1"] ** self.state["step"]))
                    # Korekta biasu dla drugiego momentu: v_t / (1 - B_2^t)
                    second_moment_estimate.divide_(1.0 - (param_group["bias_m2"] ** self.state["step"]))
                # Następnie wykonujemy właściwą aktualizację parametru
                # Przemnażamy współczynnik kroku (stepsize) przez iloraz pierwszego momentu i pierwiastka drugiego momentu powiększonego o epsilon
                # Wzór: theta = theta_{t-1} - stepsize * first_estimate / (sqrt(second_estimate) + epsilon)
                param.data.add_((-param_group["stepsize"]) * first_moment_estimate.divide_(
                    second_moment_estimate.sqrt_() + param_group["epsilon"]))
        # Zwrócenie wartości straty
        return loss
