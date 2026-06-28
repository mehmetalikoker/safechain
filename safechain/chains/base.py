from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Chain(ABC):
    """Tüm zincir türleri için ortak temel sınıf.

    Bir zincir; giriş sözlüğü alıp çıkış sözlüğü dönen, birleştirilebilir
    bir işlem birimidir. Alt sınıflar yalnızca ``invoke`` metodunu uygulamak
    zorundadır; ``run``, ``__call__`` ve ``__or__`` kolaylık katmanları bu
    temel üzerine inşa edilmiştir.
    """

    @property
    def input_keys(self) -> List[str]:
        """Zincirine beklenen giriş anahtarlarının listesi.

        Alt sınıflar bu property'yi ezerek gerçek giriş değişkenlerini
        bildirebilir. Varsayılan: boş liste.

        Returns:
            Giriş sözlüğünde bulunması gereken anahtarlar.
        """
        return []

    @property
    def output_keys(self) -> List[str]:
        """Zincirin ürettiği çıkış anahtarlarının listesi.

        Alt sınıflar bu property'yi ezerek ürettikleri çıkış değişkenlerini
        bildirebilir. Varsayılan: boş liste.

        Returns:
            Çıkış sözlüğünde yer alacak anahtarlar.
        """
        return []

    @abstractmethod
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Zinciri çalıştırır ve çıkış sözlüğünü döner.

        Args:
            inputs: Zincirin beklediği anahtar-değer çiftlerini içeren
                    giriş sözlüğü.

        Returns:
            Zincirin ürettiği çıkış anahtar-değer çiftleri.
        """

    def run(self, *args: Any, **kwargs: Any) -> str:
        """Zinciri çalıştırır ve ilk çıkış değerini string olarak döner.

        Tek positional argüman verilirse ``input_keys[0]`` anahtarına
        atanır. Birden fazla keyword argüman verilirse doğrudan giriş
        sözlüğü olarak kullanılır.

        Args:
            *args: İsteğe bağlı tek positional giriş değeri.
            **kwargs: Giriş anahtar-değer çiftleri.

        Returns:
            İlk çıkış anahtarının değeri (string).
        """
        if args and not kwargs:
            key = self.input_keys[0] if self.input_keys else "input"
            inputs: Dict[str, Any] = {key: args[0]}
        else:
            inputs = dict(kwargs)
        result = self.invoke(inputs)
        out_key = self.output_keys[0] if self.output_keys else next(iter(result))
        return result[out_key]

    def __call__(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
        """Zinciri ``chain(inputs)`` biçiminde çağrılabilir kılar.

        Giriş bir string ise ``input_keys[0]`` anahtarıyla sözlüğe
        dönüştürülür; aksi hâlde sözlük olarak kabul edilir.

        Args:
            inputs: String giriş değeri veya anahtar-değer sözlüğü.
            **kwargs: Giriş sözlüğüne eklenecek ek anahtar-değer çiftleri.

        Returns:
            ``invoke`` metodunun döndürdüğü çıkış sözlüğü.
        """
        if isinstance(inputs, str):
            key = self.input_keys[0] if self.input_keys else "input"
            inputs = {key: inputs}
        return self.invoke({**inputs, **kwargs})

    def __or__(self, other: "Chain") -> "Chain":
        """``chain_a | chain_b`` sözdizimi ile sıralı zincir oluşturur.

        İlk zincirin çıktısı ikinci zincirin girdisine otomatik aktarılır.

        Args:
            other: Sıralamada bir sonraki Chain nesnesi.

        Returns:
            İki zinciri birbirine bağlayan SimpleSequentialChain.
        """
        from safechain.chains.sequential import SimpleSequentialChain
        return SimpleSequentialChain(chains=[self, other])
