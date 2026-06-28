from __future__ import annotations
from typing import Any, Dict, List

from safechain.chains.base import Chain


class SimpleSequentialChain(Chain):
    """Bir zincirin çıktısını sıradaki zincire giriş olarak aktarır.

    Her zincirin ürettiği tüm çıkış anahtarları sonraki zincirin
    giriş havuzuna eklenir. Yalnızca tek giriş / tek çıkış zincirlerinde
    en iyi çalışır; çok değişkenli durumlar için ``SequentialChain``
    tercih edilmelidir.

    Attributes:
        chains: Sırayla çalıştırılacak Chain nesnelerinin listesi.
        verbose: ``True`` ise her zincirin girdi ve çıktısı stdout'a yazdırılır.
    """

    def __init__(self, chains: List[Chain], verbose: bool = False) -> None:
        """SimpleSequentialChain oluşturur.

        Args:
            chains: Sırayla çalıştırılacak zincirler. En az iki eleman
                    içermesi önerilir.
            verbose: Hata ayıklama çıktısı açık/kapalı. Varsayılan: False.
        """
        self.chains = chains
        self.verbose = verbose

    @property
    def input_keys(self) -> List[str]:
        """İlk zincirin beklediği giriş anahtarları.

        Returns:
            ``chains[0].input_keys`` listesi.
        """
        return self.chains[0].input_keys

    @property
    def output_keys(self) -> List[str]:
        """Son zincirin ürettiği çıkış anahtarları.

        Returns:
            ``chains[-1].output_keys`` listesi.
        """
        return self.chains[-1].output_keys

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Zincirleri sırayla çalıştırır; her çıkış bir sonrakinin girdisine eklenir.

        Args:
            inputs: İlk zincire verilecek giriş anahtar-değer çiftleri.

        Returns:
            Tüm zincirlerin çıktılarının birleştirildiği nihai durum sözlüğü.
        """
        state = dict(inputs)
        for chain in self.chains:
            if self.verbose:
                print(f"[{type(chain).__name__}] input: {state}")
            output = chain.invoke(state)
            if self.verbose:
                print(f"[{type(chain).__name__}] output: {output}")
            state.update(output)
        return state


class SequentialChain(Chain):
    """Birden fazla giriş/çıkış değişkenini destekleyen sıralı zincir.

    ``SimpleSequentialChain``'den farklı olarak, hangi değişkenlerin
    giriş kabul edildiği ve hangi değişkenlerin nihai çıkış olduğu
    açıkça belirtilir. Bu sayede karmaşık veri akışları kontrollü
    biçimde yönetilebilir.

    Attributes:
        chains: Sırayla çalıştırılacak Chain nesnelerinin listesi.
        _input_variables: Dışarıdan sağlanacak giriş değişken isimleri.
        _output_variables: Nihai çıkış olarak sunulacak değişken isimleri.
        verbose: ``True`` ise her zincirin çıktısı stdout'a yazdırılır.
    """

    def __init__(
        self,
        chains: List[Chain],
        input_variables: List[str],
        output_variables: List[str],
        verbose: bool = False,
    ) -> None:
        """SequentialChain oluşturur.

        Args:
            chains: Sırayla çalıştırılacak zincirler.
            input_variables: Kullanıcının dışarıdan sağlayacağı değişken
                             isimleri.
            output_variables: Nihai çıkış sözlüğüne dahil edilecek
                              değişken isimleri.
            verbose: Hata ayıklama çıktısı açık/kapalı. Varsayılan: False.
        """
        self.chains = chains
        self._input_variables = input_variables
        self._output_variables = output_variables
        self.verbose = verbose

    @property
    def input_keys(self) -> List[str]:
        """Kullanıcının sağlaması gereken giriş değişken isimleri.

        Returns:
            Başlangıçta dışarıdan verilmesi beklenen anahtar listesi.
        """
        return self._input_variables

    @property
    def output_keys(self) -> List[str]:
        """Nihai çıkış olarak döndürülecek değişken isimleri.

        Returns:
            Çıkış sözlüğüne dahil edilecek anahtar listesi.
        """
        return self._output_variables

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Zincirleri sırayla çalıştırır ve yalnızca istenen çıkışları döner.

        Her zincir, mevcut durumdan ihtiyacı olan anahtarları alır ve
        ürettiği çıkışları ortak havuza ekler. Sonunda yalnızca
        ``output_variables`` listesindeki anahtarlar döndürülür.

        Args:
            inputs: Kullanıcının sağladığı giriş anahtar-değer çiftleri.

        Returns:
            Yalnızca ``output_variables`` anahtarlarını içeren çıkış sözlüğü.
        """
        known = dict(inputs)
        for chain in self.chains:
            chain_inputs = {k: known[k] for k in chain.input_keys if k in known}
            output = chain.invoke(chain_inputs)
            if self.verbose:
                print(f"[{type(chain).__name__}] {output}")
            known.update(output)
        return {k: known[k] for k in self._output_variables if k in known}
