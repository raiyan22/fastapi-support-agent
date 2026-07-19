from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from pathlib import Path
import uuid

from app.db import get_db
from app.models.documents import DocumentDB, DocumentResponse
from app.rag.embed import embed_file, delete_from_chroma

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


@router.post("", response_model=DocumentResponse, status_code=201)
async def upload_document(file: UploadFile = File(...), session: AsyncSession = Depends(get_db)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    doc_id = str(uuid.uuid4())[:12].upper()
    safe_name = f"{doc_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_name

    content = await file.read()
    file_path.write_bytes(content)

    try:
        chunk_count = embed_file(str(file_path), doc_id)
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    doc = DocumentDB(
        id=None,
        filename=file.filename or safe_name,
        file_path=str(file_path),
        chunk_count=chunk_count,
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)

    return doc


@router.get("", response_model=list[DocumentResponse])
async def list_documents(session: AsyncSession = Depends(get_db)):
    result = await session.execute(select(DocumentDB).order_by(DocumentDB.uploaded_at.desc()))
    return result.scalars().all()


@router.delete("/{doc_id}")
async def delete_document(doc_id: int, session: AsyncSession = Depends(get_db)):
    result = await session.execute(select(DocumentDB).where(DocumentDB.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chroma_id = Path(doc.file_path).stem.split("_")[0]
    try:
        delete_from_chroma(chroma_id)
    except Exception:
        pass

    Path(doc.file_path).unlink(missing_ok=True)
    await session.delete(doc)
    await session.commit()

    return {"deleted": doc.filename}
