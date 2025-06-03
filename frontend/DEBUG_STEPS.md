# Document List Blank Screen Debugging Guide

## Issue
The document list page is showing a blank white screen when clicking on "Documents" or switching to the "List" view in the DocumentBrowser.

## Debugging Steps Added

### 1. Added Debug Components
- Created `DocumentListDebug.tsx` - A minimal component that logs API calls and displays debug info
- Created `MinimalTest.tsx` - A simple test component to verify React rendering

### 2. Modified Components
- Updated `DocumentBrowser.tsx` to include debug component alongside DocumentList
- Added console logging to `App.tsx` to track view changes
- Temporarily included MinimalTest component in the documents view

### 3. What to Check in Browser

1. **Open Browser DevTools Console**
   - Look for console logs starting with `[App]` and `[DocumentListDebug]`
   - Check for any error messages or warnings
   - Verify API calls are being made

2. **Check Network Tab**
   - Look for calls to `/api/v1/documents/list`
   - Verify the response status is 200
   - Check if response contains document data

3. **Check Elements/Inspector**
   - Look for the MinimalTest component (light blue background)
   - Look for the DocumentListDebug component (gray background)
   - Inspect if DocumentList is rendered but hidden

## Possible Issues and Solutions

### 1. **API Proxy Issue**
If API calls fail with 404 or CORS errors:
```bash
# Ensure both servers are running
cd backend && uvicorn api.main:app --reload --port 8000
cd frontend && npm run dev
```

### 2. **Component Rendering Issue**
If MinimalTest shows but DocumentList doesn't:
- Check for JavaScript errors in console
- Verify all imports are correct
- Check for CSS conflicts

### 3. **State Management Issue**
If the component is stuck in loading state:
- Check the debug info displayed by DocumentListDebug
- Verify the API response structure matches expected format

### 4. **CSS/Styling Issue**
If components are rendered but not visible:
- Use browser inspector to check if elements have 0 height/width
- Look for `display: none` or `visibility: hidden`
- Check z-index issues

## Quick Fixes to Try

1. **Force Refresh**
   - Hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac)
   - Clear browser cache for localhost:5173

2. **Check View Mode**
   - Click the "List" button in the DocumentBrowser header
   - The default view is "Upload", not "List"

3. **Verify Backend**
   ```bash
   curl http://localhost:8000/api/v1/documents/list?limit=20&offset=0
   ```

## To Remove Debug Code

After fixing the issue, remove the debug code:

1. Remove imports and usage of `MinimalTest` from `App.tsx`
2. Remove imports and usage of `DocumentListDebug` from `DocumentBrowser.tsx`
3. Delete the debug component files:
   - `frontend/src/components/MinimalTest.tsx`
   - `frontend/src/components/document-browser/DocumentListDebug.tsx`
   - `frontend/DEBUG_STEPS.md`

## Next Steps

1. Open the app in browser: http://localhost:5173
2. Navigate to Documents page
3. Check browser console and network tab
4. Report what you see from the debug components
5. Share any error messages or unexpected behavior