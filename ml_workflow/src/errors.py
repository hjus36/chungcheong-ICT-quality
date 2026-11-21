"""공통 예외."""


class StageError(RuntimeError):
    """단계 실행을 멈춰야 하는 오류(검증 실패, 선행 단계 누락, 규약 위반 등)."""
