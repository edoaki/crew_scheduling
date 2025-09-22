from dataclasses import dataclass


@dataclass
class StaticInitConfig:
    d_model: int = 128

    # タスク側の各サブ埋め込み次元
    d_service: int = 8
    d_direction: int = 4
    d_station_id: int = 8         # depart/arrive 両方で使用
    d_station_timepos: int = 8
    d_timepos: int = 16           # depart/arrive 時刻
    d_duration: int = 8
    d_round: int = 8
    d_flags: int = 4              # 各ブールをこの次元で埋める

    # クルー側の各サブ埋め込み次元
    d_slot_label: int = 4 
    d_signoff: int = 8
    d_window_time: int = 12       # assignable start/end

    # 語彙サイズ（必要に応じて調整）
    num_services: int = 2 # local/rapid
    num_directions: int = 2 # up/down
    num_stations: int = 6


@dataclass
class ContextConfig:
    # ベース
    d_model: int = 128

    # StationEmbedding 用
    num_stations: int = 8
    d_station_id: int = 8
    d_station_timepos: int = 16

    # crew 用のサブ埋め込み
    d_on_duty: int = 4
    d_ready_time_cyc: int = 16

    # 連続スカラー埋め込み (入力3種→出力次元)
    d_scalar_in: int = 3
    d_scalar: int = 12

    # 派生（和）のプロパティ
    @property
    def d_station_total(self) -> int:
        return self.d_station_id + self.d_station_timepos

    @property
    def crew_fuse_in(self) -> int:
        # station_emb(=id+timepos) + on_duty + ready_time_fourier + scalar_proj
        return self.d_station_total + self.d_on_duty + self.d_ready_time_cyc + self.d_scalar

