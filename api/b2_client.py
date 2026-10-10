import os
from b2sdk.v2 import B2Api, InMemoryAccountInfo


def get_b2_client():
    """
    Cria cliente B2 autenticado com as variáveis de ambiente do Cloud Run.
    """
    account_info = InMemoryAccountInfo()
    b2_api = B2Api(account_info)

    b2_api.authorize_account(
        "production",
        os.getenv("B2_KEY_ID"),
        os.getenv("B2_APP_KEY")
    )

    return b2_api


def upload_bytes(path: str, data: bytes, content_type: str):
    """
    Upload simples de bytes para o bucket Backblaze.
    """
    b2_api = get_b2_client()
    bucket = b2_api.get_bucket_by_name(os.getenv("B2_BUCKET_NAME"))

    print(f"⬆️ Upload para B2: {path} ({len(data)} bytes)")
    bucket.upload_bytes(
        data,
        file_name=path,
        content_type=content_type
    )


def download_bytes(path: str) -> bytes:
    """
    Download de bytes de um ficheiro no bucket.
    """
    b2_api = get_b2_client()
    bucket = b2_api.get_bucket_by_name(os.getenv("B2_BUCKET_NAME"))

    print(f"⬇️ Download de B2: {path}")
    file_info, file_stream = bucket.download_file_by_name(path)
    return file_stream.read()


def move_file(old_path: str, new_path: str):
    """
    Move um ficheiro dentro do bucket (copy + delete).
    """
    b2_api = get_b2_client()
    bucket = b2_api.get_bucket_by_name(os.getenv("B2_BUCKET_NAME"))

    print(f"🔁 Mover ficheiro: {old_path} → {new_path}")

    file_info, _ = bucket.download_file_by_name(old_path)

    bucket.copy(
        file_info.id_,
        new_file_name=new_path,
        destination_bucket=bucket
    )

    bucket.delete_file_version(file_info.id_, old_path)


def list_prefix(prefix: str):
    """
    Lista ficheiros com um prefixo (ex: incoming/).
    """
    b2_api = get_b2_client()
    bucket = b2_api.get_bucket_by_name(os.getenv("B2_BUCKET_NAME"))

    print(f"📂 Listar prefixo: {prefix}")
    return list(bucket.ls(prefix=prefix))


def hard_delete_file(path: str):
    """
    HARD DELETE real — remove TODAS as versões do ficheiro no B2.
    Compatível com B2SDK v2 (list_file_versions devolve FileVersion).
    """
    b2_api = get_b2_client()
    bucket = b2_api.get_bucket_by_name(os.getenv("B2_BUCKET_NAME"))

    print(f"🗑 HARD DELETE: {path}")

    versions = bucket.list_file_versions(path)
    deleted_any = False

    for file_version in versions:
        print(f"🔥 Apagar versão: {file_version.id_}")
        bucket.delete_file_version(file_version.id_, file_version.file_name)
        deleted_any = True

    if not deleted_any:
        print(f"⚠️ Nenhuma versão encontrada para hard delete: {path}")
