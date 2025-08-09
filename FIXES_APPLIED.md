# Fixes Applied to Resolve Server Issues

## Issues Found and Fixed:

### 1. Authentication Error - `'BearerTokenAuth' object has no attribute 'credentials'`
**Problem**: The `/hackrx/run` endpoint was using `Depends(get_current_user)` which expected a `BearerTokenAuth` object with a `credentials` attribute, but the middleware was handling authentication differently.

**Fix**: Updated the `get_current_user` function in `app/middleware/auth.py` to:
- Accept a `Request` object instead of `HTTPAuthorizationCredentials`
- Check if the request was already authenticated by the middleware
- Provide fallback authentication logic if needed

### 2. Startup Validation Error - `'coroutine' object has no attribute 'get'`
**Problem**: In `app/core/startup.py`, the `embedding_service.get_model_info()` method was being called without `await`, but it's an async method.

**Fix**: Added `await` to the call: `model_info = await embedding_service.get_model_info()`

### 3. Pydantic v2 Compatibility Issues
**Problem**: Several Pydantic v1 patterns were causing warnings and potential issues:
- `schema_extra` should be `json_schema_extra`
- `@validator` should be `@field_validator`
- `Config` class should be `model_config` dict

**Fixes Applied**:
- Updated all `schema_extra` to `json_schema_extra` in:
  - `app/schemas/responses.py`
  - `app/schemas/requests.py`
- Updated all `@validator` to `@field_validator` with `@classmethod` decorator in:
  - `app/schemas/requests.py`
  - `app/schemas/models.py`
  - `app/core/config.py`
- Updated `Config` classes to `model_config` dictionaries
- Fixed validator parameter access: `values.get()` → `info.data.get()`

## Files Modified:
1. `app/middleware/auth.py` - Fixed authentication dependency
2. `app/core/startup.py` - Fixed async method call
3. `app/schemas/responses.py` - Updated Pydantic v2 syntax
4. `app/schemas/requests.py` - Updated Pydantic v2 syntax
5. `app/schemas/models.py` - Updated Pydantic v2 syntax
6. `app/core/config.py` - Updated Pydantic v2 syntax

## Expected Results:
- ✅ Authentication should work properly for `/hackrx/run` endpoint
- ✅ No more 500 Internal Server Error due to credentials attribute
- ✅ Startup validation should complete without coroutine errors
- ✅ Pydantic warnings should be eliminated
- ✅ Server should start and respond to requests correctly

## Testing:
Run `python test_auth.py` to verify the authentication fix is working.